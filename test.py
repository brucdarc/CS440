
import copy as cp
import numpy as np
import random

def stateMoveTuple(state, move):
    state = cp.deepcopy(state)
    for i in range(0,3):
        state[i] = tuple(state[i])
    return (tuple(state), tuple(move))


def validMoves(state):
    results = []
    for i in range(0,3):
        for j in range(0,3):
            if i != j:
                if len(state[i]) != 0: 
                    if len(state[j]) == 0: 
                        results.append([i+1,j+1])
                    elif state[j][0] > state[i][0]: 
                        results.append([i+1,j+1])
    return results



def makeMove(state, move):
    state = cp.deepcopy(state)
    state[move[1]-1].insert(0,state[move[0]-1].pop(0))
    return state




def epsilonGreedy(epsilon, Q, state, validMovesF):
    #print("ep")
    validMoves = validMovesF(state)
    if np.random.uniform() < epsilon:
        # Random Move
        return random.choice(validMoves)
    else:
        # Greedy Move
        print("greedy")
        Qs = np.array([Q.get(stateMoveTuple(state, move), 0) for move in validMoves]) 
        return validMoves[ np.argmax(Qs) ]
    

def trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF, makeMoveF):
    
    goalState = [[], [], [1,2,3]]
    epsilon = 1.0
    Q = {}
    stepsToGoal = np.zeros(nRepetitions)
    
    for game in range(nRepetitions):
        
        step = 0
        
        state = [[1, 2, 3], [], []]
        
        done = False
        
        
        
        while not done:
            step +=1
            
            move = epsilonGreedy(epsilon, Q, state, validMovesF)
            
            newState = cp.deepcopy(state)
            
            newState = makeMoveF(newState, move)
            
            tupleSM = stateMoveTuple(state, move)
            
            if tupleSM not in Q:
                Q[tupleSM] = 0
            
            if newState == goalState:
                Q[tupleSM] = 1
                done = True
                
            if step > 1:
                TSM_old = stateMoveTuple(stateOld,moveOld)
                Q[TSM_old] += learningRate * (Q[tupleSM] - Q[TSM_old])
            
            stateOld, moveOld = state, move
            state = newState
            
            #if step%10000 == 0:
            print(str(step) + " " + str(state) + " " + str(move))
            
        stepsToGoal[game] = step
        print(str(game) + str(move))
    
    
    return Q, stepsToGoal
    

Q, stepsToGoal = trainQ(50, 0.5, 0.7, validMoves, makeMove)
print(stepsToGoal)


