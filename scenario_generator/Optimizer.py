"""///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Author(s): Nishant Marer Prabhu, Kenneth Witham
FileName: Optimizer.py
Date: 27-03-2024
Description: 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""
from scenario_generator import OptimizerHelper
import torch

"""
This function is designed to run one time with all the parameters needed to find the 
close to near optimal solution for the SNRs and Rx location
"""
def runOnce(scenario,
            scen_map,
            map_resolution,
            direction,
            los_requested,
            targetSNR,
            replicateSNR,
            txLoc,
            rxLoc,
            minDistRequirement,
            maxOutPostDist,
            mapBoundary,
            resolution,
            padding_Size, 
            kernal_Size, 
            stride_Size,
            dev,
            plotData,
            debugMode):
    
    #1. First find the minimum location of the receiver
    minimumLocationPoints = OptimizerHelper.findStartPoint(txLoc,rxLoc,minDistRequirement)
    
    #2. Find the Outpost point by extending the line between the Tx and Rx beyond its original length
    outpostPoints = OptimizerHelper.findOutPost(txLoc,rxLoc,maxOutPostDist)
    
    #3. Now verify if those new endpoints lie within the boundary of the map
    outPostPoints = OptimizerHelper.verifyOutpost(txLoc,outpostPoints,mapBoundary)
    
    #4. Spray the points between the min and Max Rx locations based on resolutions
    sprayedReceiverTensorRx = OptimizerHelper.pointSprayer(minimumLocationPoints, outPostPoints, resolution, dev)
    
    #5. replicate the Tx points to the same size as the batch size
    replicatedTxPoints = OptimizerHelper.replicatePoint(txLoc, resolution)
    
    #6. Call the channel gain function and obtain the Power and the Channel_Z values
    channel_Z, PowerDB = OptimizerHelper.channelGainCalc(replicatedTxPoints, 
                                                        sprayedReceiverTensorRx, 
                                                        scen_map, 
                                                        map_resolution,
                                                        direction,
                                                        los_requested,
                                                        scenario)
    
    #7. Squeeze the PowerDB to the appropriate shape
    PowerDB = PowerDB.squeeze(-1,-2) #128x6x1
    
    #8. Find the distance between the Tx and Rx
    completeDist = OptimizerHelper.distanceCalc(replicatedTxPoints,sprayedReceiverTensorRx)
    
    #9. Take the PowerDB and filter it, and a form a smooth curve
    filteredPowerDB = OptimizerHelper.avgFilter(PowerDB, padding_Size, kernal_Size, stride_Size)
    
    #10. Based on the filter output and using its size, clip the Tx and Rx tensors
    sprayedTensor_clipped = OptimizerHelper.clipTensor(sprayedReceiverTensorRx, filteredPowerDB)
    replicatedTxPoints_clipped = OptimizerHelper.clipTensor(replicatedTxPoints, filteredPowerDB)
    
    #11. Find the distance between the clipped Tx ad Rx
    clippedDist = OptimizerHelper.distanceCalc(replicatedTxPoints_clipped,sprayedTensor_clipped)
    
    #Convert the PowerDB and FilteredPowerDB back to SNR for plotting puposes
    filteredSNR = filteredPowerDB - scenario.noise_power_db
    PowerSNR = PowerDB - scenario.noise_power_db
    
    #12. Plot the data if the flag is set
    if plotData:
        OptimizerHelper.plotSNRvsDist(filteredPowerDB,clippedDist,PowerDB,completeDist)
        OptimizerHelper.plotSNRvsDist(filteredSNR,clippedDist,PowerSNR,completeDist)

    #Convert the FilterPower in DB to Linear Power i.e., Linear Space
    filteredPowerLinear = 10**((filteredPowerDB)/10) 

    if len(targetSNR) == 1:
        #The new check flag tells us whether we want to replicate the Power across the RxTower or distribute it
        targetPowerLinear = 10**((targetSNR + scenario.noise_power_db)/10) #SNR in dB to PowerGoal Linear Space
        
        if replicateSNR:
            #Here the SNR maybe a single variable
            #13. Using the targetPowerLinear value, find the closet possible value of the Power and determine the index, 
            # use the index for finding the near Optimal Rx location
            minTensor, maxTensor = OptimizerHelper.getMinMaxTensor(filteredPowerLinear)  
            if targetPowerLinear < torch.sum(minTensor):
                raise Exception("The TargetPower is not in the feasible SNR region of the Rx Towers")
            smallest_value, index = OptimizerHelper.findMinSNRVal(filteredPowerLinear,targetPowerLinear)
            nearOptimalRxLoc = OptimizerHelper.getMinIndexVal(index, sprayedTensor_clipped)
        else:
            #Here the powerGoalLinear maybe a single variable, However, we try to solve for total Power and distribute it across the Rx Towers
            #14. Using the powerGoalLinear value, find the closet possible value of the Power and determine the index, 
            # use the index for finding the near Optimal Rx location 
            minTensor, maxTensor = OptimizerHelper.getMinMaxTensor(filteredPowerLinear)  
            
            if targetPowerLinear < torch.sum(minTensor):
                raise Exception("The TargetPower is not in the feasible SNR region of the Rx Towers")
            
            targetPowerLinear = OptimizerHelper.distributeSNRtoRx(minTensor, maxTensor, targetPowerLinear, minTensor.shape[1], dev)
            #targetSNR = 10*torch.log10(targetPower) - scenario.noise_power_db
            #filteredPowerDB = 10*torch.log10(filteredPowerDB) - scenario.noise_power_db
            smallest_value, index = OptimizerHelper.findMinSNRVal(filteredPowerLinear,targetPowerLinear)
            nearOptimalRxLoc = OptimizerHelper.getMinIndexVal(index, sprayedTensor_clipped)
    else:
        #15. Here the SNR maybe a list
        targetPowerLinear = 10**((targetSNR + scenario.noise_power_db)/10) #SNR in dB to Linear Power
        minTensor, maxTensor = OptimizerHelper.getMinMaxTensor(filteredPowerLinear)
        
        if torch.sum(targetPowerLinear) < torch.sum(minTensor):
            raise Exception("The TargetSNR is not in the feasible SNR region of the Rx Towers")
        
        targetPowerLinear = OptimizerHelper.assignSNRtoRx(minTensor, maxTensor, targetPowerLinear, dev)
        smallest_value, index = OptimizerHelper.findMinSNRVal(filteredPowerLinear,targetPowerLinear)
        nearOptimalRxLoc = OptimizerHelper.getMinIndexVal(index, sprayedTensor_clipped)

    return channel_Z, filteredPowerLinear, nearOptimalRxLoc, targetPowerLinear

"""
The Main Optimizer function which is responsible for finding the near optimal solution
for the receiver locations and the desired SNR values
"""
def OptimalSolution(scenario,
                    scen_map,
                    map_resolution,
                    txLoc,
                    rxLoc,
                    direction="uplink",
                    los_requested=False,
                    targetSNR = 10.0,
                    replicateSNR = False,
                    minDistRequirement = 10,
                    maxOutPostDist = 100,
                    batch_size = 128,
                    padding_Size = 0, 
                    kernal_Size = 28, 
                    stride_Size= 1,
                    device = None,
                    iteration_Controller = 10,
                    errorPercentage = 1.0,
                    plotData = False,
                    debugMode = False):
    
    #assert isinstance(targetSNR, torch.Tensor), "targetSNR must be a list"
    assert rxLoc.shape[1] == len(targetSNR) or len(targetSNR) == 1, "Number of TargetSNR must be either 1 or number of receivers sent"

    mapBoundary = torch.tensor([[[0,0],[scen_map.shape[0]-1,0],[scen_map.shape[0]-1,scen_map.shape[1]-1],[0,scen_map.shape[1]-1]]]).to(device)
    
    #1. Set the flag for not found optimal Rx location
    target_Found = False
    
    #2. Replicate the Tx points to match the batch size
    replicatedTxPoints = OptimizerHelper.replicatePoint(txLoc, batch_size)
    
    #3. call the runonce function which will do all the calculations and provide the first estimate for the near optimal Rx location and SNR
    channel_Z, filteredPowerLinear, nearOptimalRxLoc, targetPowerLinear = runOnce(scenario,
                                                                        scen_map,
                                                                        map_resolution,
                                                                        direction,
                                                                        los_requested,
                                                                        targetSNR,
                                                                        replicateSNR,
                                                                        txLoc,
                                                                        rxLoc,
                                                                        minDistRequirement,
                                                                        maxOutPostDist,
                                                                        mapBoundary,
                                                                        batch_size,
                                                                        padding_Size, 
                                                                        kernal_Size, 
                                                                        stride_Size,
                                                                        device,
                                                                        plotData,
                                                                        debugMode)
    
    iteration_val = 0
    
    #4. Now replicate the nearoptimal Rx
    replicatedRxLoc = nearOptimalRxLoc.repeat((batch_size,1,1))
    
    #5. start a while loop
    while(True):
        #6. Call the channel gain function
        channel_Z, PowerDB = OptimizerHelper.channelGainCalc(replicatedTxPoints, 
                                                            replicatedRxLoc, 
                                                            scen_map, 
                                                            map_resolution,
                                                            direction,
                                                            los_requested,
                                                            scenario)
        
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
        smallest_value, index = OptimizerHelper.findMinSNRVal(nearOptimalSNR,targetSNR)
        
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
            # if debugMode: print("I am Unable to find the optimal solution, please retry with a new scenario set!!")
            # raise Exception("I am Unable to find the optimal solution, please retry with a new scenario set!!")
            return target_Found, None, None, None
        
        iteration_val += 1