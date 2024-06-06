"""///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Author(s): Nishant Marer Prabhu, Kenneth Witham
FileName: OptimierHelper.py
Date: 27-03-2024
Description: 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""
from __future__ import division 
import torch
import matplotlib.pyplot as plt
import numpy as np
from ortools.graph.python import linear_sum_assignment
from ortools.linear_solver import pywraplp
from scipy.optimize import minimize

"""
txLoc: Transmittor location (x,y,h_ut)
rxLoc: Receiver location (x,y,h_bs)
minDistRequirement: Minimum distance between the transmittor and receiver

This function finds the minimum location of the receiver from the transmitter
"""
def findStartPoint(txLoc,rxLoc,minDistRequirement):

    if len(txLoc.shape) == len(rxLoc.shape):
        if txLoc.shape[2] == rxLoc.shape[2]:
            totalDistance = torch.cdist(txLoc, rxLoc, p=2)
            distanceRatio = (minDistRequirement/totalDistance).squeeze()
            
            #Find the new start point which allows for mindistance requirement
            minimumLocationPoints = torch.stack((((1 - distanceRatio) * txLoc[:,:,0] + distanceRatio * rxLoc[:,:,0]),
                                                 ((1 - distanceRatio) * txLoc[:,:,1] + distanceRatio * rxLoc[:,:,1]), 
                                                 (rxLoc[:,:,2])), dim = 2)
            
            return minimumLocationPoints
    else:
        raise ValueError("The shape of the points must either be [1,1,3] or have the same dimensions")
      
"""
txLoc: Transmittor location (x,y,h_ut)
rxLoc: Receiver location (x,y,h_bs)
maxOutPostDistance: distance beyond the receiver for the maximum point location of receiver along
                    the vector of tx and rx
This function finds the maximum location of the receiver from the transmitter
"""
def findOutPost(txLoc,rxLoc,maxOutPostDistance):

    if len(txLoc.shape) == len(rxLoc.shape):
        if txLoc.shape[2] == rxLoc.shape[2]:
            
            txRxDist = torch.cdist(txLoc, rxLoc, p=2)
            extendedDistance = (float(maxOutPostDistance)/100.0) * txRxDist + txRxDist
            distanceRatio = (extendedDistance/txRxDist).squeeze()
            
            outpostPoint = torch.stack((((1 - distanceRatio) * txLoc[:,:,0] + distanceRatio * rxLoc[:,:,0]),
                                        ((1 - distanceRatio) * txLoc[:,:,1] + distanceRatio * rxLoc[:,:,1]), 
                                        (rxLoc[:,:,2])), dim = 2)
            
            return outpostPoint
    else:
        raise ValueError("The shape of the points must either be [1,1,3] or have the same dimensions")

"""
mapBoundaryA: (x,y) location of the map boundary
mapBoundaryB: (x,y) location of the map boundary collinear to mapBoundaryA
txLoc: Transmittor location (x,y,h_ut)
outPostLoc: Receiver location (x,y,h_bs) (Possible location of the receiver)

This function ensures that the outpostloc of the receivers are within the map boundary,
and if not, it finds a new point on the boundary
"""
def getPointOfIntersection(mapBoundaryA,mapBoundaryB,txLoc,outPostLoc):
    
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

"""
mapBoundaryA: (x,y) location of the map boundary
mapBoundaryB: (x,y) location of the map boundary collinear to mapBoundaryA
txLoc: Transmittor location (x,y,h_ut)
outPostLoc: Receiver location (x,y,h_bs) (Possible location of the receiver)

This function given the outpost locations, loops through the boundaries of the map
ensuring the receivers are within the map boundary
"""
def verifyOutpost(txLoc,outPostPoints,mapBoundryPoints):
    
    if len(txLoc.shape) == len(outPostPoints.shape):
        if txLoc.shape[2] == outPostPoints.shape[2]:
            
            for i in range(mapBoundryPoints.shape[1]):
                boundaryPointA, boundaryPointB = (mapBoundryPoints[:,i].reshape(1,1,-1), mapBoundryPoints[:,(i + 1) % mapBoundryPoints.shape[1]].reshape(1,1,-1))
                outPostPoints = getPointOfIntersection(boundaryPointA,boundaryPointB,txLoc,outPostPoints)
    
    return outPostPoints

"""
startLoc: Point where to start the point spray
endLoc: point where to stop the point spray
resolution: number of points in between the start and endpoint, evenly distributed
return: an evenly spaced combination of (x,y,h_bs points along the line of startLoc and endLoc
"""
def pointSprayer(startLoc, endLoc, resolution, dev):
    # x = (x2-x1)*(n random number)+x1
    # y = (y1-y2)/(x1-x2)*(x-x1) + y1
    
    spray = torch.linspace(0, 1, resolution).to(dev).reshape(-1, 1, 1)
    Ones = torch.ones(resolution).to(dev).reshape(-1,1,1)
    
    x = (endLoc[:,:,0] - startLoc[:,:,0]).reshape(1,-1,1) * spray + startLoc[:,:,0].reshape(1, -1, 1)
    y = (startLoc[:,:,1] - endLoc[:,:,1]).reshape(1,-1,1)/(startLoc[:,:,0]-endLoc[:,:,0]).reshape(1,-1,1)*(x - startLoc[:,:,0].reshape(1, -1, 1)) + startLoc[:,:,1].reshape(1, -1, 1)
    z = Ones * endLoc[:,:,2].reshape(1,-1,1)
    sprayedTensor = torch.stack((x,y,z),dim=2).squeeze(-1)
    
    return sprayedTensor

"""
point: a tensor of shape [batchsize, number of elements, degree of each element]
resolution: number of times to be repeated 

This function replicates a given tensor by the given resolution
"""
def replicatePoint(point, resolution):    
    return point.repeat(resolution,1,1)    

"""
txLocations: Transmitter location  tensor (x,y,h_ut)
rxLocations: Receiver location tensor (x,y,h_bs)
scen_map
map_resolution
direction
los_requested
scenario
"""
def channelGainCalc(txLocations, 
                    rxLocations, 
                    scen_map, 
                    map_resolution,
                    direction,
                    los_requested,
                    scenario):
    
    scenario.update_topology(txLocations, rxLocations, scen_map, map_resolution=map_resolution, direction=direction, los_requested=los_requested)
    z, rx_pow = scenario.generate_channels()

    #We will convert the power to Power DB, this is to ensure the filter can operate
    PowerDB = 10*torch.log10(rx_pow)

    return z, PowerDB

"""
Power: a tensor which needs to be filtered
padding_size: number of zeros to be added on either side of the tensor
kernal_size: filter size
stride_Size: filter movement size
"""
def avgFilter(Power, padding_Size, kernal_Size, stride_Size):
    pad_op = torch.nn.ReflectionPad1d(padding_Size)
    average_Filter = torch.nn.AvgPool1d(kernel_size = kernal_Size,
                                        padding = 0,
                                        stride = stride_Size, count_include_pad=False)
    
    filteredList = []
    for i in range(Power.shape[1]):
        filteredList.append(average_Filter(pad_op(Power[:,i,:].reshape(1,1,-1))))
    
    filtered_Power = torch.cat(filteredList, axis=1).permute(2,1,0)
    
    return filtered_Power

"""
SNRs: a tensor containing the generated SNRs
targetSNR: desired SNR to acheive

This function finds the difference between the target and given SNR tensor
"""
def findMinSNRVal(SNRs,targetSNR):
    
    #percentage_difference = ((torch.abs(targetSNR - SNRs)) / abs(targetSNR)) * 100.0
    #smallest_value, index = torch.min(percentage_difference, 0)
    if SNRs.dim() == 0:
        rms_errors = torch.sqrt((targetSNR - SNRs)**2)
        smallest_value, index = torch.min(rms_errors, 0)
        return smallest_value, index
    elif SNRs.dim() == 1:
        rms_errors = torch.sqrt((targetSNR - SNRs.unsqueeze(0))**2)
        smallest_value, index = torch.min(rms_errors, 0)
        return smallest_value, index
    else:
        rms_errors = torch.sqrt(torch.mean((targetSNR - SNRs)**2, dim=2))
        smallest_value, index = torch.min(rms_errors, 0)
        return smallest_value.view(-1,1), index.view(-1,1)
    #return smallest_value, index

"""
index: a tensor of index values where near optimal value occures
dataTensor: a tensor where using the index tensor, the actual data points need to be extracted

This function uses the index values to extract the required data from the data tensor
"""
def getMinIndexVal(index, dataTensor):
    tensors = []
    for i in range(dataTensor.shape[1]):
        tensors.append(dataTensor[:,i,:].unsqueeze(1)[index[i]])
    result = torch.cat(tensors, axis=1)
    
    return result

"""
filteredSNR: a tensor containing the filtered SNR values
dist1: a tensor containing distance between each Tx and Rx
unfilteredSNR: a tensor containing unfiltered SNR values
dist2: a tensor containing distance between each Tx and Rx

This function takes variable sizes of SNRs, hence the two input values for distances between Tx and Rx
It plots the filtered and unfiltered SNR overlapping each other
"""
def plotSNRvsDist(filteredSNR,dist1,unfilteredSNR,dist2):
    
    for i in range(dist1.shape[1]):
        fig = plt.figure(figsize=(8,8))
        fSNRindividual = filteredSNR[:,i,:].squeeze().tolist()
        unfSNRindividual = unfilteredSNR[:,i,:].squeeze().tolist()
        d_clipped = dist1[:,i].tolist()
        d_unclipped = dist2[:,i].tolist()
        
        plt.plot(d_unclipped,unfSNRindividual)
        plt.plot(d_clipped,fSNRindividual)
        plt.savefig(f"plt1_{i}.jpg")
    
"""
tensorToModify: a tensor who's dimension needs to be clipped
refTensor: tensor for reference

This function uses the shape of the reference tensor and clips the other
"""
def clipTensor(tensorToModify, refTensor):
    
    missingPoints = tensorToModify.shape[0] - refTensor.shape[0]
    if missingPoints % 2 == 0:
        numPointsToClipLeft = missingPoints // 2
        numPointsToClipRight = missingPoints - numPointsToClipLeft
    else:
        numPointsToClipLeft = missingPoints // 2
        numPointsToClipRight = missingPoints - numPointsToClipLeft
    
    if numPointsToClipLeft != 0 and numPointsToClipRight != 0:
        clipped_tensor = tensorToModify[numPointsToClipLeft:-numPointsToClipRight, :, :]
    else:
        clipped_tensor = tensorToModify
    
    if clipped_tensor.shape[0] == refTensor.shape[0]:
        return clipped_tensor
    else:
        raise ValueError("The shape does not match")
    
"""
tensor1
tensor2
This function takes two tensor of same shape and finds the l2 distance between them
"""
def distanceCalc(tensor1,tensor2):
    totalDistance = torch.cdist(tensor1[:,:,0:2], tensor2[:,:,0:2], p=2)
    return totalDistance.squeeze(1)

"""
This function takes in a tensor and returns the min and max values
"""
def getMinMaxTensor(givenTensor):
    minTensor = givenTensor.min(dim=0, keepdim = True)[0]
    maxTensor = givenTensor.max(dim=0, keepdim = True)[0]
    return minTensor, maxTensor

"""
minTensor: a tensor containing the minimum values
maxTensor: a tensor containing the maximum values
targetPower: a tensor containing the target Power values to be assigned to Rx
This function uses the linear assignment algorithm to determine which Receivers is able to achieve the given Power values
"""
def assignSNRtoRx(minTensor, maxTensor, targetPower, dev):

    newPowerTensor = torch.zeros(minTensor.shape,device=dev)
    targetPower = targetPower.reshape(1,len(targetPower),1).to(dev)
    meanPower = (maxTensor + minTensor) / 2.0
    distance_Vector = torch.abs(targetPower - meanPower.transpose(0, 1)).squeeze().transpose(0, 1).to('cpu').numpy()
    RxNodes_set, Power_Nodes_set = np.meshgrid(np.arange(distance_Vector.shape[1]), np.arange(distance_Vector.shape[0]))

    Power_nodes = Power_Nodes_set.ravel()
    Rx_nodes = RxNodes_set.ravel()
    arc_costs = distance_Vector.ravel()

    Assign_SNRtoRx = linear_sum_assignment.SimpleLinearSumAssignment()
    Assign_SNRtoRx.add_arcs_with_cost(Power_nodes, Rx_nodes, arc_costs)
    status = Assign_SNRtoRx.solve()

    if status == Assign_SNRtoRx.OPTIMAL:
        for i in range(0, Assign_SNRtoRx.num_nodes()):
            newPowerTensor[:,Assign_SNRtoRx.right_mate(i)] = targetPower[:,i]
    else:
        raise Exception("Unable to assign Power to Rx Nodes")

    return newPowerTensor

"""
minTensor: a tensor containing the minimum values
maxTensor: a tensor containing the maximum values
targetSNR: a tensor containing the target SNR values to be assigned to Rx
numOfRxTowers: Number of receivers
dev: Device on which the tensors are located
This function is LP solver, it tries to satisfy the constraints provided. It does not have an actual optimization equation to solve, 
hence, its job is to just meet the requirements of constraints and provide an output.
"""
def GLOPSolver(minTensor, maxTensor, targetPower, numOfRxTowers, dev):

    targetPower = targetPower.to('cpu').tolist()[0]
    newPowerTensor = torch.zeros(minTensor.shape,device=dev)
    LPSolver = pywraplp.Solver.CreateSolver("GLOP")

    #Enable it to solve problems in two directions i.e., the dual of the problem
    LPSolver.SetSolverSpecificParametersAsString("use_dual_simplex:true")
    
    variables = []
    for i in range(numOfRxTowers):
        var = LPSolver.NumVar(-1 * (LPSolver.infinity()), LPSolver.infinity(), f'x{i+1}')
        variables.append(var)
        LPSolver.Add(var >= float(minTensor[:,i,0].to('cpu').numpy()[0]))
        LPSolver.Add(var <= float(maxTensor[:,i,0].to('cpu').numpy()[0]))

    #Add constraint that the sum of variables must be equal to the target SNR, hence the lower and upper bound is target value
    constraint = LPSolver.Constraint((targetPower), (targetPower))
    for var in variables:
        constraint.SetCoefficient(var, 1)

    #Here we do not specify an objective function, however every solver does need something, hence we pass a dummy objective here
    objective = LPSolver.Objective()
    for var in variables:
        objective.SetCoefficient(var, 0)
    objective.SetMinimization()
     
    # Solve the problem
    status = LPSolver.Solve()

    if status == LPSolver.OPTIMAL:
        for i, var in enumerate(variables):
            newPowerTensor[:,i] = var.solution_value()
        return newPowerTensor
    return None

"""
minTensor: a tensor containing the minimum values
maxTensor: a tensor containing the maximum values
targetPower: a tensor containing the target Power values to be assigned to Rx
numOfRxTowers: Number of receivers
dev: Device on which the tensors are located
This function does solves a quadratic equation i.e., a least square estimation technique.
We provide an inital guess and here for simplicity, we use the average of the Minimum and Maximum SNR achievable by the Rx Towers.
We also use the Min and Max SNR as constraints, to ensure we stay within the limit of SNR achievable
"""
def ObjectiveFunction(individualParamaters_list, targetSum):
    return (sum(individualParamaters_list) - targetSum)**2

def PowellSolver(minTensor, maxTensor, targetPower, numOfRxTowers, dev):
    
    targetPower = targetPower.to('cpu').tolist()[0]
    newPowerTensor = torch.zeros(minTensor.shape,device=dev)
    minTensorFlattened = minTensor.flatten().to('cpu').numpy()
    maxTensorFlattened = maxTensor.flatten().to('cpu').numpy()

    constraints_Bounds = list(zip(minTensorFlattened, maxTensorFlattened))
    
    Initial_Guess = [(np.mean(pair) * 1.0) for pair in constraints_Bounds]
    OptimalSolutionResult = minimize(ObjectiveFunction, Initial_Guess, args=(targetPower,), method='Powell', bounds=constraints_Bounds)
    
    if OptimalSolutionResult.success:
        if OptimalSolutionResult.fun < 1e-5:            
            for i, x in enumerate(OptimalSolutionResult.x):
                newPowerTensor[:,i] = x
            return newPowerTensor
    return None

"""
minTensor: a tensor containing the minimum values
maxTensor: a tensor containing the maximum values
targetPower: a tensor containing the target Power values to be assigned to Rx
numOfRxTowers: Number of receivers
dev: Device on which the tensors are located

To provide redundancy, as there cases when one of the solvers may not converge. In those cases, the function calls
another optimizer. Here we have the GLOPsolver i.e., the LP solver and the Powell Solver which is quadratic solver.
If neither of them achieve the goal, an exception is raised.
"""
def distributeSNRtoRx(minTensor, maxTensor, targetPower, numOfRxTowers, dev):
    
    scaleTarget = 0
    #Scale the parameters
    if (minTensor < 1e-7).all():
        minTensor = minTensor * 1e12
    if (minTensor > 1e7).all():
        minTensor = minTensor * 1e-12

    if (maxTensor < 1e-7).all():
        maxTensor = maxTensor * 1e12
    if (maxTensor > 1e7).all():
        maxTensor = maxTensor * 1e-12
    
    if targetPower < 1e-7:
        targetPower = targetPower * 1e12
        scaleTarget = 1e12
    if targetPower > 1e7:
        targetPower = targetPower * 1e-12
        scaleTarget = 1e-7


    powerTensor = GLOPSolver(minTensor, maxTensor, targetPower, numOfRxTowers, dev)
    if powerTensor is not None:
        return (powerTensor / scaleTarget)
    
    powerTensor = PowellSolver(minTensor, maxTensor, targetPower, numOfRxTowers, dev)
    if powerTensor is not None:
        return (powerTensor / scaleTarget)
    else:
        raise Exception("Optimization failed, neither of the solvers are able to converge")