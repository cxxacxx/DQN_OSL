#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:46:00 2019

@author: cxx
"""
import pygame 
import numpy as np
import keras
import math
import DQN_OSL
import Plume
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy.io as scio

WIDTH = 800             #width of game screen
HEIGHT = 800            #height of game screen
BLACK = (0,0,0)         #background color
ROBOT = (255,255,255)   #snake color
SOURCE = (255,0,0)        #ball color
WALL = (255,255,0)      #head color (not used in default version)
rowSize = 80            #size of each row in the game
columnSize = 80         #size of each column in the game
numChannels = 4         #trajectory, mean_concentration, wind_direction, map
numRobots = 1           #number of robots
walls=list()

epsilon = 1           #initial epsilon ie. exploration rate
epsilonDecayRate = 0.0002   #by how much we decrease epsilon each epoch 0.0002
minEpsilon = 0.001      #minimum of epsiol 0.001

nbEpoch = 300000        #on how many epochs model is trained on
learningRate = 0.0001   #learning rate of the model
memSize = 60000         #Deep Q Learning memory size
gamma = 0.9             #gamma parameter which defines by how much next state influences previous state
batchSize = 32          #size of batch neural network is trained on every iteration (every move) 

defaultReward = -0.5    #the default reward for every action
negativeReward = -10     #the reward for hitting itself or wall
positiveReward = 20      #the reward for finding the source

#filepathToOpen = 'model_epoch_wall_35000.h5'

train =True      #if we want to train our model then we set it to True if we want to test a pretrained model we set it to False
resultsFolder='results/DQNa'
nRows = int(HEIGHT/rowSize)
nColumns = int(WIDTH/columnSize)
scrn = np.zeros((nRows, nColumns)) 
robotTra = np.zeros((nRows, nColumns))
concentration = np.zeros((nRows, nColumns))
windDirection = np.zeros((nRows, nColumns))

fail = False

success = False
bth = 1

if not os.path.isdir(resultsFolder):
	os.makedirs(resultsFolder)

class robot(object):
    #def __init__(self, iS = (100,100,3), nO = 3, lr = 0.0005):
    def __init__(self, iS = (100,100,3), nO = 3, lr = 0.0005):
        self.learningRate = lr
        self.inputShape = iS
        self.numOutputs = nO
        self.model = Sequential() 
        
        self.model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = self.inputShape))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(64, (2,2), activation = 'relu'))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(units = 256, activation = 'relu'))
        
        self.model.add(Dense(units = self.numOutputs))
        self.model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = self.learningRate))
        #plot_model(self.model, to_file='model.png',show_shapes = True)
    def loadModel(self, fp):
        self.model = load_model(fp)
        return self.model
    
def windDirectionMeasure():
    val = -math.pi
    return val

def concentrationMeasure(isTrapped,isSide):
    C_expected = plume.expectedHit(currentRobotPos[0]-sourcex,currentRobotPos[1]-sourcey,isTrapped,isSide)
    C_expected = round(C_expected)
    C_measured = np.random.poisson(C_expected)
    #print("expected:",C_expected,"measured:",C_measured)
    return C_measured

def drawWall():
    global scrn
    walls.clear()
    numWalls = np.random.randint(0,8)
    rnd = (np.random.randint(0,nRows), np.random.randint(0,nColumns))
    for i in range(numWalls):
        if rnd[1]+i > 9:
            break
        scrn[rnd[0]][rnd[1]+i] = 1
        walls.append([rnd[0],rnd[1]+i]) 
    
def isTrapped(robotPos,sourcePos):
    if walls:
        if (robotPos[0]>=walls[0][0]) & (sourcex < walls[0][0]):
            y = robotPos[1]
            if (y>=walls[0][1])&(y<=walls[-1][1]):
                return 1
    return 0

def isSide(robotPos,sourcePos):
    if walls:
        if (robotPos[0]>=walls[0][0]) & (sourcex < walls[0][0]):
            y = robotPos[1]
            if (y==walls[0][1]-1)|(y==walls[-1][1]+1):
                return 1
        if (robotPos[0]==walls[0][0]-1) & (sourcex < walls[0][0]):
            y = robotPos[1]
            if (y>=walls[0][1]-1)&(y<=walls[-1][1]+1):
                return 1
    return 0
      
def drawRobot(direction):
    global currentRobotPos
    global fail
    global success
    global scrn
    global robotTra
    global windDirection
    global concentration
    
    if direction == -1:
        scrn = np.zeros((nRows, nColumns))
        drawWall()
        robotTra = np.zeros((nRows, nColumns))
        concentration = np.zeros((nRows, nColumns))
        windDirection = np.zeros((nRows, nColumns))        
        while(1):
            rnd1 = np.random.randint(0,3)
            if (rnd1 == 0):
                rnd = (np.random.randint(0,4), np.random.randint(4,9))
            if (rnd1 == 1):
                rnd = (np.random.randint(6,9), np.random.randint(4,9))
            if (rnd1 == 2):
                rnd = (np.random.randint(5,9), np.random.randint(0,2))
            rnd = (np.random.randint(0,nRows), np.random.randint(0,nColumns))
            if scrn[rnd[0]][rnd[1]] ==1:
                continue
            else:
                break

        scrn[rnd[0]][rnd[1]] = 2
        currentRobotPos = [rnd[0], rnd[1]]
        robotTra[currentRobotPos[0]][currentRobotPos[1]] = 1
    elif direction == 1:
        x = currentRobotPos[0]
        y = currentRobotPos[1]
        if x - 1 >= 0:
            if scrn[x-1][y]==1 :
                fail = True
                return
            scrn[currentRobotPos[0]][currentRobotPos[1]] = 0
            currentRobotPos[0] -= 1
            robotTra[x-1][y] += 1
            scrn[currentRobotPos[0]][currentRobotPos[1]] = 2
        else:
            fail = True
    elif direction == 2:
        x = currentRobotPos[0]
        y = currentRobotPos[1]
        if y + 1 < nColumns :
            if scrn[x][y+1]==1:
                fail = True
                return
            scrn[currentRobotPos[0]][currentRobotPos[1]] = 0
            currentRobotPos[1] += 1
            robotTra[x][y+1] += 1
            scrn[currentRobotPos[0]][currentRobotPos[1]] = 2
        else:
            fail = True
    elif direction == 3:
        x = currentRobotPos[0]
        y = currentRobotPos[1]
        if y - 1 >= 0:
            if scrn[x][y-1]==1 :
                fail = True
                return
            scrn[currentRobotPos[0]][currentRobotPos[1]] = 0
            currentRobotPos[1] -= 1
            robotTra[x][y-1] += 1
            scrn[currentRobotPos[0]][currentRobotPos[1]] = 2
        else:
            fail = True
    elif direction == 4:
        x = currentRobotPos[0]
        y = currentRobotPos[1]
        if x + 1 < nRows :
            if scrn[x+1][y]==1:
                fail = True
                return
            scrn[currentRobotPos[0]][currentRobotPos[1]] = 0
            currentRobotPos[0] += 1
            robotTra[x+1][y] += 1
            scrn[currentRobotPos[0]][currentRobotPos[1]] = 2
        else:
            fail = True

def measure():
    global success
    global windDirection
    global concentration
    robotIsTrapped = 0#isTrapped(currentRobotPos,[sourcex,sourcey])
    robotIsSide = 0#isSide(currentRobotPos,[sourcex,sourcey])
    wind = windDirection[currentRobotPos[0]][currentRobotPos[1]]
    wind = (wind *(robotTra[currentRobotPos[0]][currentRobotPos[1]]-1) + windDirectionMeasure())/robotTra[currentRobotPos[0]][currentRobotPos[1]]
    windDirection[currentRobotPos[0]][currentRobotPos[1]] = wind
    
    con = concentration[currentRobotPos[0]][currentRobotPos[1]]
    con = (con *(robotTra[currentRobotPos[0]][currentRobotPos[1]]-1) + concentrationMeasure(robotIsTrapped,robotIsSide))/robotTra[currentRobotPos[0]][currentRobotPos[1]]
    concentration[currentRobotPos[0]][currentRobotPos[1]] = con
    if currentRobotPos in [[sourcex-1,sourcey],[sourcex+1,sourcey],[sourcex,sourcey-1],[sourcex,sourcey+1]] :
        success = True
        
def drawSource():
    while(1):
        rnd = (np.random.randint(0,nRows), np.random.randint(0,nColumns))
        if scrn[rnd[0]][rnd[1]] != 0 :
            continue
        else:
#            if (rnd[0]-1>=0)&(scrn[rnd[0]-1][rnd[1]] == 2):
#                continue
#            if (rnd[0]+1<=9)&(scrn[rnd[0]+1][rnd[1]] == 2):
#                continue
#            if (rnd[1]-1>=0)&(scrn[rnd[0]][rnd[1]-1] == 2):
#                continue
#            if (rnd[1]+1<=9)&(scrn[rnd[0]][rnd[1]+1] == 2):
#                continue
#            else:
#                break
            break
            
    return rnd[0],rnd[1]
    
def mapArray():
    for row in range(nRows):
        for column in range(nColumns):
            x = column * columnSize
            y = row * rowSize
            if scrn[row][column] == 2:
                pygame.draw.rect(screen, ROBOT, (x + bth,y + bth,columnSize - (bth*2),rowSize - (bth*2)))
            if scrn[row][column] == 1:
                pygame.draw.rect(screen, WALL, (x + bth,y + bth,columnSize - (bth*2),rowSize - (bth*2)))
            if (row == sourcex) & (column == sourcey) :
                pygame.draw.rect(screen, SOURCE, (x + bth,y + bth,columnSize - (bth*2),rowSize - (bth*2)))
            
                
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('OSL')
screen.fill(BLACK)
direction = 1
start = True

robot = robot((nRows, nColumns, numChannels), 4, learningRate)
model = robot.model
model = robot.loadModel(filepathToOpen)

    #plot_model(model, to_file='model10x10.png',show_shapes = True)
reward = 0.0
nActions = 0
dqn = DQN_OSL.DQN(memSize, gamma)
score = 0

results = []
nEpochsTotreward = 0.
maxQValue = []
lastMove = direction
#f=open('reward','w+')

for epoch in range(nbEpoch):
    plume = Plume.model(diffusion = 0.1, decay = 0.004, width = 1,scaling = np.random.randint(40,40))
    nActions = 0
    fail = False
    success = False
    currentState = np.zeros((1,nRows,nColumns,1))
    nextState = np.zeros((1,nRows,nColumns,1))
    drawRobot(-1)
    sourcex, sourcey = drawSource()
#    np.savetxt(runFolder+'/source',[sourcex,sourcey])
    measure()
    currentState = np.concatenate((np.array(robotTra.reshape((1,nRows,nColumns,1))), np.array(concentration.reshape((1,nRows,nColumns,1))), np.array(windDirection.reshape((1,nRows,nColumns,1))), np.array(scrn.reshape((1,nRows,nColumns,1)))), axis = 3)
    nextState = np.concatenate((np.array(robotTra.reshape((1,nRows,nColumns,1))), np.array(concentration.reshape((1,nRows,nColumns,1))), np.array(windDirection.reshape((1,nRows,nColumns,1))), np.array(scrn.reshape((1,nRows,nColumns,1)))), axis = 3)
    loss = 0.0
    reward = 0.0
    direction = 1
    score = 0
    qValue = 0.
    rqv = 0.
#    np.savetxt(runFolder+'/map',scrn)
    while not (fail | success):
        nActions += 1
        if start:
            reward = defaultReward
            Qv = model.predict(currentState)[0]
            rqv += np.max(Qv)
            if (np.random.rand() <= epsilon and train):
                action = np.random.randint(0, 4) + 1
            else:            
                action = np.argmax(Qv) + 1
            direction = action
            if nActions == 100:
                fail = True
                       
        if start:
            drawRobot(direction)
            measure()
            mapArray()

        if success: 
            reward = positiveReward
#            f=open(runFolder+'/result','w')
#            f.write('success')
#            f.close()
            
        if fail:
            reward = negativeReward
#            f=open(runFolder+'/result','w')
#            f.write('fail')
#            f.close()
        
        score +=reward
        
        nextState = np.concatenate((np.array(robotTra.reshape((1,nRows,nColumns,1))), np.array(concentration.reshape((1,nRows,nColumns,1))), np.array(windDirection.reshape((1,nRows,nColumns,1))), np.array(scrn.reshape((1,nRows,nColumns,1)))), axis = 3)
        dqn.remember([currentState, action - 1, reward, nextState], success | fail)
        
            
            
        lastMove = direction
        currentState = np.copy(nextState)
    
        pygame.display.flip()     
        if train:
            pygame.time.wait(0)
        else:
            pygame.time.wait(0)
        screen.fill(BLACK)
        if train: 
            inputs,targets = dqn.getBatch(model, batchSize)
            loss += model.train_on_batch(inputs, targets)
    #np.savetxt(runFolder+'/trajectory',robotTra)        
    if epsilon > minEpsilon+0.0001:
        epsilon -= epsilonDecayRate
    if nActions == 0:
        nActions = 0.0001
    qValue += rqv / nActions
    nEpochsTotreward += score
    if epoch % 20 == 0 and epoch != 0:
        if epoch % 20 == 0 and epoch != 0 and train:
            model.save('model_room_epoch_'+str(epoch)+'.h5')
        
        results.append(nEpochsTotreward / 100)
        #f.write(str(nEpochsTotreward / 100)+'\n')
        #f.flush()
        plt.plot(results)
        plt.ylabel('Average Score')
        plt.xlabel('Epoch / 100')
        plt.show()
        maxQValue.append(qValue/100)
        plt.plot(maxQValue)
        plt.ylabel('Avg Max Q Value')
        plt.xlabel('Epoch / 100')
        plt.show()
        qValue = 0.
        nEpochsTotreward = 0.
    print('Avg Loss: ' + str(round(loss / nActions,3)) + ' Epoch: ' + str(epoch) + ' Reward: ' + str(score) + ' Epsilon: ' + str(round(epsilon,5)))
#    
#f.close()   
