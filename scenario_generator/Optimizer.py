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
    
    #6. Call the channel gain function and obtain the SNRs and the Channel_Z values
    channel_Z, SNRs = OptimizerHelper.channelGainCalc(replicatedTxPoints, 
                                                        sprayedReceiverTensorRx, 
                                                        scen_map, 
                                                        map_resolution,
                                                        direction,
                                                        los_requested,
                                                        scenario)
    
    #7. Squeeze the SNRs to the appropriate shape
    SNRs = SNRs.squeeze(-1,-2) #128x6x1
    
    #8. Find the distance between the Tx and Rx
    completeDist = OptimizerHelper.distanceCalc(replicatedTxPoints,sprayedReceiverTensorRx)
    
    #9. Take the SNRs and filter it, and a form a smooth curve
    filteredSNR = OptimizerHelper.avgFilter(SNRs, padding_Size, kernal_Size, stride_Size)
    
    #10. Based on the filter output and using its size, clip the Tx and Rx tensors
    sprayedTensor_clipped = OptimizerHelper.clipTensor(sprayedReceiverTensorRx, filteredSNR)
    replicatedTxPoints_clipped = OptimizerHelper.clipTensor(replicatedTxPoints, filteredSNR)
    
    #11. Find the distance between the clipped Tx ad Rx
    clippedDist = OptimizerHelper.distanceCalc(replicatedTxPoints_clipped,sprayedTensor_clipped)
    
    #12. Plot the data if the flag is set
    if plotData:
        OptimizerHelper.plotSNRvsDist(filteredSNR,clippedDist,SNRs,completeDist)
    
    #13. Using the targetSNR value, find the closet possible value of the SNR and determine the index, 
    # use the index for finding the near Optimal Rx location
    smallest_value, index = OptimizerHelper.findMinSNRVal(filteredSNR,targetSNR)
    nearOptimalRxLoc = OptimizerHelper.getMinIndexVal(index, sprayedTensor_clipped)

    return channel_Z, filteredSNR, nearOptimalRxLoc

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
                    minDistRequirement = 10,
                    maxOutPostDist = 100,
                    batch_size = 128,
                    padding_Size = 0, 
                    kernal_Size = 27, 
                    stride_Size= 1,
                    device = None,
                    iteration_Controller = 10,
                    errorPercentage = 1.0,
                    plotData = False,
                    debugMode = False):
    
    mapBoundary = torch.tensor([[[0,0],[scen_map.shape[0],0],[scen_map.shape[0],scen_map.shape[1]],[0,scen_map.shape[1]]]]).to(device)
    
    #1. Set the flag for not found optimal Rx location
    target_Found = False
    
    #2. Replicate the Tx points to match the batch size
    replicatedTxPoints = OptimizerHelper.replicatePoint(txLoc, batch_size)
    
    #3. call the runonce function which will do all the calculations and provide the first estimate for the near optimal Rx location and SNR
    channel_Z, filteredSNR, nearOptimalRxLoc = runOnce(scenario,
                                                        scen_map,
                                                        map_resolution,
                                                        direction,
                                                        los_requested,
                                                        targetSNR,
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
        channel_Z, SNRs = OptimizerHelper.channelGainCalc(replicatedTxPoints, 
                                                            replicatedRxLoc, 
                                                            scen_map, 
                                                            map_resolution,
                                                            direction,
                                                            los_requested,
                                                            scenario)
        
        #7. Reduce the dimensions of the SNR as per the dimension requirement
        SNRs = SNRs.squeeze(-1,-2)
        
        #8. Find the index of the closest value of the SNR to the targetSNR
        smallest_value, index = OptimizerHelper.findMinSNRVal(SNRs,targetSNR)
        
        #9. Use the index to determine the optimal SNR values
        nearOptimalSNRs = OptimizerHelper.getMinIndexVal(index, SNRs)
        nearOptimalChannel_Z = OptimizerHelper.getMinIndexVal(index, channel_Z)
        
        if debugMode: print("Current iter: ",iteration_val)
        if debugMode: print("Current small value: ",smallest_value.squeeze().tolist())
        
        if (smallest_value <= errorPercentage).all():
            target_Found = True
            if debugMode: print('The Target is found!!')
            if debugMode: print("The near Optimal Rx locations are: ", nearOptimalRxLoc)
            if debugMode: print("The near Optimal Channel_Z are: ", nearOptimalChannel_Z)
            if debugMode: print("The near Optimal SNRs are: ", nearOptimalSNRs)
            
            return target_Found, nearOptimalRxLoc, nearOptimalChannel_Z, nearOptimalSNRs
        
        if iteration_val == iteration_Controller:
            # if debugMode: print("I am Unable to find the optimal solution, please retry with a new scenario set!!")
            raise Exception("I am Unable to find the optimal solution, please retry with a new scenario set!!")
        
        iteration_val += 1