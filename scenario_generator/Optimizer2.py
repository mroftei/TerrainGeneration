from typing import Any
import torch
import matplotlib.pyplot as plt
from sionna_torch import SionnaScenario

from ortools.graph.python import linear_sum_assignment
from ortools.linear_solver import pywraplp
from scipy.optimize import minimize


class ChannelGenerator:
    def __init__(
        self,
        map_resolution = 1,
        n_rx = 1,
        n_tx = 1,
        frame_size = 1024,
        f_c = .92e9,
        bw = 30e3,
        noise_power_dB = None,
        direction = "uplink",
        replicateSNR = False,
        minDist = 1,
        batch_size = 128,
        max_iters = 10,
        max_error = 0.1,
        seed = 42,
        dtype = torch.float32,
        device = None,
        debug = False,
        *args: Any,
        **kwds: Any
    ) -> None:
        self.config = locals()
        self.device = device

        self.sionna = SionnaScenario(n_bs=n_rx, n_ut=n_tx, batch_size=batch_size, n_time_samples=frame_size, f_c=f_c, bw=bw, noise_power_dB=noise_power_dB, seed=seed, dtype=dtype.to_complex(), device=device)
        self.avg_filter = torch.nn.Sequential(
            torch.nn.ReplicationPad1d(27//2),
            torch.nn.AvgPool1d(kernel_size=27, stride=1)
        )

    def __call__(
        self, 
        scenario_map,
        tx_xyz,
        rx_xyz,
        target_snr,
        los_requested=False,
        *args: Any, 
        **kwds: Any
    ) -> Any:
        assert tx_xyz.shape[0] == 1 and tx_xyz.shape[1] == self.config['n_tx'], "Tx shape incorrect"
        assert rx_xyz.shape[0] == 1 and rx_xyz.shape[1] == self.config['n_rx'], "Rx shape incorrect"
        assert rx_xyz.shape[1] == len(target_snr) or len(target_snr) == 1, "Length of target_snr must be either 1 or number of receivers sent"

        # Replicate the Tx points to match the batch size
        # tx_xyz = tx_xyz.repeat(self.config['batch_size'],1,1) 

        target_pow_linear = 10**((target_snr + self.sionna.noise_power_db)/10) #SNR in dB to PowerGoal Linear Space

        power_est_db, rx_xyz_sprayed = self._estimate_path_gain(tx_xyz, rx_xyz, scenario_map, los_requested)

        rx_near_optimal = self._get_initial_guess(target_pow_linear, power_est_db, rx_xyz_sprayed)
        

    def _estimate_path_gain(self, tx_xyz, rx_xyz, scen_map, los_requested):
        map_size = scen_map.shape[0]
        
        #1. First find the minimum location of the receiver
        distance_path = torch.cdist(tx_xyz, rx_xyz, p=2)
        distance_ratio = (self.config['minDist']/distance_path).reshape(1,-1,1)
        min_points = torch.cat((((1 - distance_ratio) * tx_xyz[:,:,:2] + distance_ratio * rx_xyz[:,:,:2]), 
                                (rx_xyz[:,:,2:])), dim = 2)
        
        # TODO: make this dumber and faster by just taking closest point to boundary and still inside
        #2. Find the Outpost point by extending the line between the Tx and Rx beyond its original length
        extended_distance = (float(map_size)/100.0) * distance_path + distance_path
        distance_ratio = (extended_distance/distance_path).reshape(1,-1,1)
        max_points = torch.cat((((1 - distance_ratio) * tx_xyz[:,:,:2] + distance_ratio * rx_xyz[:,:,:2]), 
                                (rx_xyz[:,:,2:])), dim = 2)
        
        #3. Now verify if those new endpoints lie within the boundary of the map
        map_boundary = torch.tensor([[[0,0],[map_size-1,0],[map_size-1,map_size-1],[0,map_size-1]]], device=self.device)
        for i in range(map_boundary.shape[1]):
            boundaryPointA, boundaryPointB = (map_boundary[:,i].reshape(1,1,-1), map_boundary[:,(i + 1) % map_boundary.shape[1]].reshape(1,1,-1))
            max_points = self._get_poi(boundaryPointA,boundaryPointB,tx_xyz,max_points)
        
        #4. Spray the points between the min and Max Rx locations based on resolutions
        tx_xyz_replicated = tx_xyz.repeat(self.config['batch_size'],1,1) 
        spray = torch.linspace(0, 1, self.config['batch_size'], device=self.device).reshape(-1, 1, 1)
        ones = torch.ones((self.config['batch_size'], 1, 1), device=self.device)
        x = (max_points[:,:,0] - min_points[:,:,0]).reshape(1,-1,1) * spray + min_points[:,:,0].reshape(1, -1, 1)
        y = (min_points[:,:,1] - max_points[:,:,1]).reshape(1,-1,1)/(min_points[:,:,0]-max_points[:,:,0]).reshape(1,-1,1)*(x - min_points[:,:,0].reshape(1, -1, 1)) + min_points[:,:,1].reshape(1, -1, 1)
        z = ones * max_points[:,:,2].reshape(1,-1,1)
        sprayed_tensor = torch.stack((x,y,z),dim=2).squeeze(-1)
        
        #5. Call the channel gain function and obtain the Power and the Channel_Z values
        self.sionna.update_topology(tx_xyz_replicated, sprayed_tensor, scen_map, map_resolution=self.config['map_resolution'], direction=self.config['direction'], los_requested=los_requested)
        h_T, rx_pow = self.sionna.generate_channels()
        rx_pow_db = 10*torch.log10(rx_pow)
        h_T = h_T.to(self.device)
        rx_pow_db = rx_pow_db.to(self.device).squeeze((2,4)) #128,n_rx,x_tx  

        #8. Find the distance between the Tx and Rx
        # totalDistance = torch.cdist(tx_xyz[:,:,0:2], sprayed_tensor[:,:,0:2], p=2).squeeze(1)
        
        #9. Take the PowerDB and filter it, and a form a smooth curve
        # for i in range(Power.shape[1]):
        #     filteredList.append(average_Filter(pad_op(Power[:,i,:].reshape(1,1,-1))))
        # filtered_Power = torch.cat(filteredList, axis=1).permute(2,1,0)
        filtered_power_db = self.avg_filter(rx_pow_db.permute((2,1,0))).permute((2,1,0))
        
        #10. Based on the filter output and using its size, clip the Tx and Rx tensors
        # sprayedTensor_clipped = OptimizerHelper.clipTensor(sprayedReceiverTensorRx, filteredPowerDB)
        # replicatedTxPoints_clipped = OptimizerHelper.clipTensor(replicatedTxPoints, filteredPowerDB)
        
        #Convert the PowerDB and FilteredPowerDB back to SNR for plotting puposes
        filtered_snr_db = filtered_power_db - self.sionna.noise_power_db
        rx_snr_db = rx_pow_db - self.sionna.noise_power_db
        
        #12. Plot the data if the flag is set
        if self.config['debug']:
            clipped_dist = torch.cdist(tx_xyz_replicated[:,:,0:2], sprayed_tensor[:,:,0:2], p=2)
            self.plotSNRvsDist(filtered_snr_db, clipped_dist,rx_snr_db,clipped_dist)
        
        return filtered_power_db, sprayed_tensor

    def _get_initial_guess(self, target_pow_linear, power_est_db, rx_xyz_sprayed):
        #Convert the FilterPower in DB to Linear Power i.e., Linear Space
        power_est_linear = 10**((power_est_db)/10) 
        
        assert target_pow_linear > torch.sum(power_est_linear.min(0)[0]), "The TargetPower is not in the feasible SNR region of the Rx Towers"
        
        #The new check flag tells us whether we want to replicate the Power across the RxTower or distribute it
        if self.config['replicateSNR']:
            #Here the SNR maybe a single variable
            #13. Using the targetPowerLinear value, find the closet possible value of the Power and determine the index, 
            # use the index for finding the near Optimal Rx location
            target_pow_linear_perchan = target_pow_linear.repeat(self.config['n_rx'])
        else:
            #Here the powerGoalLinear maybe a single variable, However, we try to solve for total Power and distribute it across the Rx Towers
            #14. Using the powerGoalLinear value, find the closet possible value of the Power and determine the index, 
            # use the index for finding the near Optimal Rx location 
                
            # Scale the parameters between 1e6:1e-6
            new_max, new_min = 1.0, 1e-6
            scale = ((new_max - new_min) / (power_est_linear.max() - power_est_linear.min()))
            scale_bias = new_min - power_est_linear.min() * scale

            target_pow_range = 10**((10*torch.log10(target_pow_linear) + torch.tensor([-0.5, +0.5], device=self.device))/10)
            target_pow_linear_perchan = self._glop_solver(target_pow_range*scale+scale_bias, power_est_linear*scale+scale_bias)
            target_pow_linear_perchan = (target_pow_linear_perchan-scale_bias)/scale
            
            # powerTensor = PowellSolver(minTensor, maxTensor, targetPower, numOfRxTowers, dev)
            # if powerTensor is not None:
            #     return (powerTensor / scaleTarget)
            # else:
            #     raise Exception("Optimization failed, neither of the solvers are able to converge")
        
        rms_errors = torch.sqrt((target_pow_linear_perchan[None,:,None] - power_est_linear)**2)
        index = rms_errors.argmin(0)
        nearOptimalRxLoc = torch.gather(rx_xyz_sprayed, 0, index[None].repeat((1,1,3)))

        return nearOptimalRxLoc

    def _gen_matching_channels(self, ):
        #4. Now replicate the nearoptimal Rx
        replicatedRxLoc = nearOptimalRxLoc.repeat((batch_size,1,1))
        
        #5. start a while loop
        while(True):
            #6. Call the channel gain function
            self.sionna.update_topology(tx_xyz_replicated, sprayed_tensor, scen_map, map_resolution=self.config['map_resolution'], direction=self.config['direction'], los_requested=los_requested)
            h_T, rx_pow = self.sionna.generate_channels()
            rx_pow_db = 10*torch.log10(rx_pow)
            h_T = h_T.to(self.device)
            rx_pow_db = rx_pow_db.to(self.device).squeeze((2,4)) #128,n_rx,x_tx  
            
            #7. Reduce the dimensions of the PowerDB as per the dimension requirement and change to Linear Space
            LinearPower = 10**((PowerDB.squeeze(-1,-2))/10)
            
            #8. Find the index of the closest value of the Power to the targetPowerLinear
            if len(targetSNR) == 1:
                smallest_value, index = OptimizerHelper.findMinSNRVal(LinearPower,targetPowerLinear)
                nearOptimalPower = OptimizerHelper.getMinIndexVal(index, LinearPower)
                if replicateSNR:    
                    nearOptimalSNR = 10*torch.log10(nearOptimalPower) - scenario.noise_power_db
                else:
                    nearOptimalSNR = 10*torch.log10(nearOptimalPower.sum()) - scenario.noise_power_db
            else:
                smallest_value, index = OptimizerHelper.findMinSNRVal(LinearPower,targetPowerLinear)
                nearOptimalPower = OptimizerHelper.getMinIndexVal(index, LinearPower)
                nearOptimalSNR = (10*torch.log10(nearOptimalPower) - scenario.noise_power_db).view(1,-1)
            
            #9. Use the index to determine the optimal SNR values
            nearOptimalChannel_Z = OptimizerHelper.getMinIndexVal(index, channel_Z)
            smallest_value, index = OptimizerHelper.findMinSNRVal(nearOptimalSNR.squeeze(),targetSNR)
            
            if debugMode: print("Current iter: ",iteration_val)
            if debugMode: print("Current small value: ",smallest_value.squeeze().tolist())
            
            if (smallest_value <= errorPercentage).all():

                target_Found = True
                if debugMode: print('The Target is found!!')
                if debugMode: print("The near Optimal Rx locations are: ", nearOptimalRxLoc)
                if debugMode: print("The near Optimal Channel_Z are: ", nearOptimalChannel_Z)
                if debugMode: print("The near Optimal SNRs are: ", nearOptimalSNR)
                
                return target_Found, nearOptimalRxLoc, nearOptimalChannel_Z, nearOptimalSNR
            
            if iteration_val == iteration_Controller:
                return target_Found, None, None, None
            
            iteration_val += 1

    def  _get_poi(self, mapBoundaryA, mapBoundaryB, txLoc, outPostLoc):
        Det = ((outPostLoc[:,:,1] - txLoc[:,:,1]) * (mapBoundaryB[:,:,0] - mapBoundaryA[:,:,0]) - 
            (outPostLoc[:,:,0] - txLoc[:,:,0]) * (mapBoundaryB[:,:,1] - mapBoundaryA[:,:,1]))
        
        nonZeroIndices = (Det == 0)
        Det[nonZeroIndices] += torch.finfo(torch.float64).eps
        
        t = ((outPostLoc[:,:,0] - txLoc[:,:,0])*(mapBoundaryA[:,:,1]-txLoc[:,:,1]) - 
            (outPostLoc[:,:,1] - txLoc[:,:,1])*(mapBoundaryA[:,:,0]-txLoc[:,:,0]))/Det
        
        tIndex = ((t > 0) & (t < 1))
        
        u = ((mapBoundaryB[:,:,0] - mapBoundaryA[:,:,0])*(mapBoundaryA[:,:,1]-txLoc[:,:,1]) - 
            (mapBoundaryB[:,:,1] - mapBoundaryA[:,:,1])*(mapBoundaryA[:,:,0]-txLoc[:,:,0]))/Det
        
        uIndex = ((u > 0) & (u < 1))
        
        completeIndex = (tIndex & uIndex)

        x = mapBoundaryA[:,:,0] + t * (mapBoundaryB[:,:,0] - mapBoundaryA[:,:,0])
        y = mapBoundaryA[:,:,1] + t * (mapBoundaryB[:,:,1] - mapBoundaryA[:,:,1])
        z = outPostLoc[:,:,2]
        newPoints = torch.stack((x,y,z),dim=2)
        
        outPostLoc[completeIndex] = newPoints[completeIndex]
        
        return outPostLoc

    def _glop_solver(self, target_power_range, power_est_linear):
        newPowerTensor = torch.zeros(power_est_linear.shape[1], device=self.device)
        LPSolver = pywraplp.Solver.CreateSolver("GLOP")

        #Enable it to solve problems in two directions i.e., the dual of the problem
        # LPSolver.SetSolverSpecificParametersAsString("use_dual_simplex:true")
        
        #Add constraint that the sum of variables must be equal to the target SNR, hence the lower and upper bound is target value
        constraint = LPSolver.Constraint((target_power_range[0].item()), (target_power_range[1].item()))

        #Here we do not specify an objective function, however every solver does need something, hence we pass a dummy objective here
        objective = LPSolver.Objective()
        objective.SetMinimization()

        variables = []
        for i in range(power_est_linear.shape[1]):
            var = LPSolver.NumVar(-1 * (LPSolver.infinity()), LPSolver.infinity(), f'x{i+1}')
            LPSolver.Add(var >= float(power_est_linear[-1,i,0].item()))
            LPSolver.Add(var <= float(power_est_linear[0,i,0].item()))
            variables.append(var)
            constraint.SetCoefficient(var, 1)
            objective.SetCoefficient(var, 0)

        # Solve the problem
        status = LPSolver.Solve()

        assert status == LPSolver.OPTIMAL, "No Optimal solution possible"
        for i, var in enumerate(variables):
            newPowerTensor[i] = var.solution_value()
        return newPowerTensor
            

    def plotSNRvsDist(self, filteredSNR, dist1, unfilteredSNR, dist2):
        for i in range(dist1.shape[1]):
            fig = plt.figure(figsize=(8,8))
            fSNRindividual = filteredSNR[:,i,:].squeeze().tolist()
            unfSNRindividual = unfilteredSNR[:,i,:].squeeze().tolist()
            d_clipped = dist1[:,i].tolist()
            d_unclipped = dist2[:,i].tolist()
            
            plt.plot(d_unclipped,unfSNRindividual)
            plt.plot(d_clipped,fSNRindividual)
            plt.savefig(f"plt1_{i}.jpg")