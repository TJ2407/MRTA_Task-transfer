from random import randint, choice
import yaml
import time as tim
import itertools
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import math
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
import pickle
# import matplotlib.pyplot as plt
w=[]
# %%
def sp(input_list):
    # Check if the input list has at least two elements
    if len(input_list) < 2:
        raise ValueError("Input list must have at least two elements.")

    # Use a list comprehension to create pairs
    pairs = [[input_list[i], input_list[i + 1]] for i in range(len(input_list) - 1)]
    return pairs
def calculate_distance(coord1, coord2):
    # Calculate Euclidean distance between two coordinates
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5

def total_distance(path_pretransfer):
    # Calculate the total distance of the path_pretransfer
    distance = 0
    for i in range(len(path_pretransfer) - 1):
        #print(path_pretransfer[i],path_pretransfer[i+1])
        distance += calculate_distance(path_pretransfer[i], path_pretransfer[i + 1])
    return distance

def find_shortest_path(coords1, start_point=None, additional_points=None):
    coords=[start_point]
    s=[]
    d=[]
    if(additional_points!=None):
        for i in additional_points:
            coords+=[i]
    for i in coords1:
     coords+=i
     s+=[i[0]]
     d+=[i[1]]
    n = len(coords)
    min_distance = float('inf')
    shortest_path = []
    tot=1000000
    minp=0
    # Generate all possible permutations of the coordinates
    for perm in permutations(coords):
        path_pretransfer = list(perm)
        for i in range(len(s)):
            if(path_pretransfer.index(start_point)==0):
             if path_pretransfer.index(s[i])<path_pretransfer.index(d[i]):
                if(total_distance(path_pretransfer)<=tot):
                    tot=total_distance(path_pretransfer)
                    minp=path_pretransfer
                    return minp
    
    return None
    
    
def piecewise_linear(x, points):
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if x1 <= x <= x2 or x2 <= x <= x1:
            slope = (y2 - y1) / (x2 - x1)
            return slope * (x - x1) + y1

    return None

def find_intersection(initial_path_pretransfer, final_path_post_transfer):
    intersections = []

    for i in range(len(initial_path_pretransfer) - 1):
        x1, y1 = initial_path_pretransfer[i]
        x2, y2 = initial_path_pretransfer[i + 1]

        for j in range(len(final_path_post_transfer) - 1):
            x3, y3 = final_path_post_transfer[j]
            x4, y4 = final_path_post_transfer[j + 1]

            # Check if the segments are not parallel
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

            if denominator != 0:
                x_intersect = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
                x_intersect /= denominator

                y_intersect = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
                y_intersect /= denominator

                # Check if x-coordinate is not too close to zero
                if abs(x_intersect) > 1e-15:
                    intersections.append((x_intersect, y_intersect))

    return intersections
import math
def generate_piecewise_function(coordinates):
    if len(coordinates) < 2:
        return "Not enough points to form line segments."

    functions = []

    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]

        # Check if the line segment is vertical
        if x1 == x2:
            functions.append({
                'domain': (x1,),
                'range': (y1, y2),
                'equation': f'y = {min(y1, y2)} <= y <= {max(y1, y2)}'
            })
        else:
            slope = (y2 - y1) / (x2 - x1)
            y_intercept = y1 - slope * x1

            functions.append({
                'domain': (x1, x2),
                'range': (y1, y2),
                'equation': f'y = {slope} * x + {y_intercept} for {x1} <= x <= {x2}'
            })

    return functions




def classify_point_by_equations_with_points(point, piecewise_functions):
    x, y = point

    for i, section in enumerate(piecewise_functions):
        x1, y1 = section['domain'][0], section['range'][0]
        x2, y2 = section['domain'][-1], section['range'][-1]

        if x1 == x2:
            # Handle vertical line segments
            if x == x1 and min(y1, y2) <= y <= max(y1, y2):
                return [[x1, y1],[x2,y2]]
        else:
            # Check if the point lies on the line segment using mx + c equation
            slope = (y2 - y1) / (x2 - x1)
            y_intercept = y1 - slope * x1

            if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and y == slope * x + y_intercept:
                return [[x1, y1],[x2,y2]]
    return 0


# Example usage:

path_coordinates = [(0, 0), (1, 1), (2, 0), (3, 2), (2, 1)]
piecewise_functions = generate_piecewise_function(path_coordinates)
for i, section in enumerate(piecewise_functions):
    print(f"Section {i + 1}:")
    print(f"Domain: {section['domain']}")
    print(f"Range: {section['range']}")
    print(f"Equation: {section['equation']}")
    print()
point_to_classify = (2.5, 1.5)
classification_result = classify_point_by_equations_with_points(point_to_classify, piecewise_functions)

print(classification_result)

def d(p1,p2):
    dis=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return dis
def piecewise_length(path_pretransfer, point1, point2):
    length = 0
    p1=0
    p2=0
    in1=0
    in2=0
   
    pwf=generate_piecewise_function(path_pretransfer)
    cl1 = classify_point_by_equations_with_points(point1, pwf)
    cl2= classify_point_by_equations_with_points(point2, pwf)
    print(cl1,cl2)
    print(point1,point2,path_pretransfer)
    p1=cl1[1]
    p2=cl2[0]
    print(p1,p2,path_pretransfer)
    in1=path_pretransfer.index((p1))
    in2=path_pretransfer.index((p2))
    if in2+1!=in1:
     length=d(p1,point1)+d(p2,point2)
     print("l1",length)
     tr=path_pretransfer[in1:in2+1]
     tr1=path_pretransfer[in1:in2]
     for y in tr1:
         print(tr.index(y)+1,tr)
         length+=(d(y,tr[tr.index(y)+1]))
        # Check if the segment intersects with the line segment between point1 and point2
    else:
        length=d(point1,point2) 
    return length
def chk(c,xc,yc):
    f=1
    for i in range(len(xc)):
        if c.index(xc[i])>c.index(yc[i]):
            f=0
    if f==1:
        return True
    else:
        return False

def ideal_intersection_point(initial_path_pretransfer, final_path_post_transfer,vel1,vel2,intersection):
    # Find intersection points between paths
    

    # Calculate robot velocities
    robvel = [vel1, vel2]

    # Initialize variables for the best intersection point
    best_intersection_point = None
    best_time_difference = float('inf')

    for point_i in intersection:
        for point_j in intersection:
            # Calculate the time difference for each robot
            time_differences = [
                abs(point_i[0] - point_j[0]) / robvel[0],
                abs(point_i[1] - point_j[1]) / robvel[1]
            ]
            max_time_difference = max(time_differences)

            if max_time_difference < best_time_difference:
                best_time_difference = max_time_difference
                best_intersection_point = point_i  # Choose any point as they all have the same time difference

    return best_intersection_point
def dist(point1, point2):
    distance = math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return distance
def euc(inp):
    inpl = sp(inp)
    sigma=0
    
    for i in inpl :
       d=math.sqrt((i[0][0]-i[1][0])**2+(i[0][1]-i[0][1])**2)
       sigma=sigma+d
    return sigma
with open('path_dir.dat', 'rb') as file:
    path_dir = pickle.load(file)
print(path_dir)
print()
import math
def opt(c,r,r1):
    distas={}
    distad={}
    if(r1 != None):
     myl=[r1]
     d1=math.sqrt((r1[0]-r[0])**2+(r1[1]-r[1])**2)
     distas[d1]=r1
    else:
     myl=[]
    l=[]
    print("c", c)
    for i in c:
        myl.append(i[0])
        d1=math.sqrt((i[0][0]-r[0])**2+(i[0][1]-r[1])**2)
        myl.append(i[1])
        d2=math.sqrt((i[1][0]-r[0])**2+(i[1][1]-r[1])**2)
        distas[d1]=i[0]
        distad[d2]=i[1]

    xc=[]
    yc=[]
    for i in c:
        xc.append(i[0])
        yc.append(i[1])
    prob=0
    
    for i in range(len(myl)):
        if i ==0:
            prob=distas.copy()
        print(myl)
        print(list(prob.keys()))
        tempr=min(prob.keys())
        p=prob[tempr]
        l.append(p)
        del prob[tempr]
        if(p in xc):
            y=yc[xc.index(p)]
            k=list(distad.keys())
            v=list(distad.values())
            prob[k[v.index(y)]]=y
    print(l)
    return(l)
    '''l=list(itertools.permutations(myl))
    minindx=0
    eucmin=euc(l[0])
    for i in l:
        if(chk(i,xc,yc)):
            if(euc(i)<=eucmin):
                minindx=l.index(i)
                eucmin=euc(i)
    return l[minindx]'''
with open('red_coordinates.dat', 'rb') as file:
  r= pickle.load(file)
print(r)


def perplen(point,initial_path_pretransfer,final_path_post_transfer,vel1,vel2):
    perpd=10000
    per1=10000
    per2=10000
    po=[]
    po1=[]
    po2=[]
    for i in final_path_post_transfer :
        for j in range(2):
            #d=dist(point[1],i)
            d=math.sqrt((point[1][0]-i[j][0])**2+(point[1][1]-i[j][1])**2)
            if d <= perpd:
                perpd=d
                po=i[j].copy()
    for i in initial_path_pretransfer:
        for j in range(2):
            if(point[1][0]!=i[j][0] and point[1][1]!=i[j][1]):
                d= math.sqrt((point[1][0]-i[j][0])**2+(point[1][1]-i[j][1])**2)
            
                d2=math.sqrt((point[0][0]-i[j][0])**2+(point[0][1]-i[j][1])**2)
                if d <= per1:
                    per1=d
                    po1=i[j].copy()
                if d2 <= per2:
                    per2=d
                    po2=i[j].copy()
    perpd/=vel2
    per1/=vel1
    per2/=vel1

    return [perpd+per2,per1+per2,po]
def rfy(l):
    e=[]
    for i in l:
        e.append(r.index(i))
    return e

def pfind(los):
   
    for i in path_dir:
        
        if list(i[0])==los[0] and list(i[1])==los[1]:
            return (i[2])
def ffp(intermediate_points):
    # Check if there are at least two intermediate points
    if len(intermediate_points) < 2:
        raise ValueError("At least two intermediate points are required.")

    # Initialize the final path_pretransfer
    fp = []
    #print(intermediate_points)
    ip=sp((intermediate_points))
    fp=[(intermediate_points)[0]]
    #print(ip)
    for i in ip:
        #print(i)
        pf=pfind(i)
        #print(pf)
        for ii in pf:
         fp.append(ii)
    return fp
class TD_task:
    taskID = 0
    arrivalTime = 0
    startTime = 0# any time after this
    finishTime = 0 # any time before this
    pickup = ""
    destination = ""
    demand = 0
    type = 0 #depends on robot
    timeconstraint = 0 #0 for soft, 1 for hard
    softPenalty = 10 #perSecond
    hardPenalty = 1000000000
with open("/Users/tejas_sriganesh/Downloads/distance_params.yaml", 'r') as file1:
    points= yaml.safe_load(file1)
    points = points["world_nodes"]
print(points)
# dist= np.loadtxt('/Users/tejas_sriganesh/Desktop/dist.txt', usecols=range(22))
# dist = np.round(dist, decimals=3)
dist=[]
for g in points:
    d=[]
    for h in points:
        l=len(pfind([[points[g]['x'],points[g]['y']],[points[h]['x'],points[h]['y']]]))
        print(l)
        d.append(l)
    dist.append(tuple(d))
places = {}
z=0
print(dist)
for i in points:
    places[i] = z
    # print(places[i])
    z+=1
# %%
TIMETOCHARGE = 300
class Robot:
    def getNearestDock(self, pos):
        global dist
        asd = 10e9
        for i in ["dock_1", "dock_2", "dock_3"]:
            asd  = min(asd, dist[places[i]][places[pos]])        
        return asd
    
    def getCentralDock(self, start, next):
        global dist
        asd = 10e9
        for i in ["dock_1", "dock_2", "dock_3"]:
            asd  = min(asd, dist[places[i]][places[start]]+dist[places[i]][places[next]])    
        return asd


    def tempSTN(self, nodes, completionTime, time, pos, taskI, energyRem, started, capacityRem, currList, currPen, bufferSize, usedEnergy):
        global dist
        global places
        
        #print(currList)
        
        if len(taskI)==0:
            # print(currList)
            if(currPen < self.minPen):
                self.minPen = currPen
                t = currList.copy()
                self.retList=t
                self.minCT = completionTime
                # self.minCT = min(self.minCT, currList[-1][1])
                self.minTT = min(self.minTT, currList[-2][1])
                self.minUsedEnergy = max(self.minUsedEnergy, usedEnergy)
            # retList.append(t)
            # print(retList)
            
        else:
            for i in taskI:
                #print(currList)
                if(i in started):   # Has the task been picked up
                    # print(time)
                    completionTime = time + (dist[places[pos]][places[nodes[i].destination]])/self.velocity  #Current Time Update
                    timecpy = completionTime
                    usedEnergy = (0.5*(self.mass+self.capacity-capacityRem)*self.velocity**2)*((dist[places[pos]][places[nodes[i].destination]])/self.velocity)
                    enRem = energyRem-(0.5*(self.mass+self.capacity-capacityRem)*self.velocity**2)*((dist[places[pos]][places[nodes[i].destination]])/self.velocity) #Current energy remaining update
                    # print(enRem)
                    prevPos = pos
                    pos = nodes[i].destination
                    #print([prevPos,pos,"d",self.robotID])                                  #current Position
                    # print(str(i)+"drop", timecpy)
                    newcapacityRem = capacityRem + nodes[i].demand                   # New capacity
                    newBuff = bufferSize-1
                    cap = self.capacity-newcapacityRem
                    tt = (enRem*1.0)/(0.5*(self.mass+cap)*(self.velocity**2))             # possible maximum total travel time with new capacity
                
                    currList.append([pos,  completionTime, timecpy, usedEnergy,self.robotID,nodes[i].finishTime-nodes[i].arrivalTime])                           
                    taskIcpy = taskI.copy()
                    taskIcpy.remove(i)            # remove curr task from task list as it has been delivered
                    if((completionTime + self.getNearestDock(pos)/self.velocity)>tt):              # Does robot run out of energy before reaching nearest dock
                        # print("energy over")
                        # tempSTN(nodes, time, pos, set(), energyRem, started, capacityRem, currList, 10e9)
                        completionTime = time + TIMETOCHARGE + (self.getCentralDock(prevPos, pos)/self.velocity)
                        timecpy = completionTime - TIMETOCHARGE
                        usedEnergy = usedEnergy + (0.5*(self.mass+cap)*(self.velocity**2))
                        if(nodes[i].timeconstraint ==0):
                            tempPen = currPen + max(0, (completionTime-nodes[i].finishTime))   #penalty update for soft task
                            # timecpy = completionTime
                        else:                                                        #Hard task penalty update
                            if(completionTime-nodes[i].finishTime>0):
                                tempPen =currPen+10e9
                            else:
                                tempPen = currPen
                        self.tempSTN(nodes, completionTime, timecpy, pos, taskIcpy, self.energy, started, newcapacityRem, currList, tempPen, newBuff, usedEnergy)   #Recursion with current time, current pos, current energy rem

                    else:
                        if(nodes[i].timeconstraint ==0):
                            tempPen = currPen+max(0, (completionTime-nodes[i].finishTime))   #penalty update for soft task
                        else:                                                        #Hard task penalty update
                            if( completionTime-nodes[i].finishTime>0):
                                tempPen =currPen+10e9
                            else:
                                tempPen = currPen
                        self.tempSTN(nodes, completionTime, timecpy, pos, taskIcpy, enRem, started, newcapacityRem, currList, tempPen, newBuff,  usedEnergy)   #Recursion with current time, current pos, current energy rem

                    currList.pop()
                else:       # if task not picked up
                    # print(pos)
                    # print(places[pos])
                    # print(dist[places[pos]][places[nodes[i].pickup]])
                    if(bufferSize > 0):
                        # print(time)
                        completionTime = time+(dist[places[pos]][places[nodes[i].pickup]])/self.velocity #Current Time Update
                        # print(completionTime)
                        timecpy = completionTime
                        enRem = energyRem-(0.5*(self.mass+self.capacity-capacityRem)*self.velocity**2)*((dist[places[pos]][places[nodes[i].pickup]])/self.velocity) #Current energy remaining update
                        usedEnergy = (0.5*(self.mass+self.capacity-capacityRem)*self.velocity**2)*((dist[places[pos]][places[nodes[i].pickup]])/self.velocity)
                        # print(enRem)
                        prevPos = pos
                        pos = nodes[i].pickup 
                        #print([prevPos,pos,"p",self.robotID])  #current pos
                        if(nodes[i].startTime > timecpy): # Basically wait for task to start
                             
                             completionTime = nodes[i].startTime
                        currList.append([pos,  completionTime, timecpy, usedEnergy,self.robotID,nodes[i].finishTime-nodes[i].arrivalTime])
                        #print(currList)
                        startedCpy = started.copy()
                        startedCpy.add(i)
                        # print(nodes[i].demand)
                        newcapacityRem = capacityRem-nodes[i].demand
                        # print(nodes[i].demand, newcapacityRem)
                        cap = self.capacity-capacityRem
                        newBuff = bufferSize-1
                        if(newcapacityRem <= 0):
                                self.tempSTN(nodes, completionTime, time, pos, set(), enRem, started, self.capacity, currList, 10e9, newBuff, usedEnergy)
                        else:
                            # print(0.5*(self.mass+cap)*(self.velocity**2))
                            tt = (enRem*1.0)/(0.5*(self.mass+cap)*(self.velocity**2))        # possible travel time with new capacity              
                            if(( completionTime+self.getNearestDock(pos)/self.velocity)>tt): # Does robot run out of energy before reaching nearest dock
                                # print("energy over")
                                # tempSTN(nodes, time, pos, set(), energyRem, started, capacityRem, currList, 10e9)
                                completionTime = time + TIMETOCHARGE + (self.getCentralDock(prevPos, pos)/self.velocity)
                                usedEnergy = (0.5*(self.mass+cap)*(self.velocity**2)) * (self.getCentralDock(prevPos, pos)/self.velocity)
                                timecpy =  completionTime - TIMETOCHARGE
                                self.tempSTN(nodes, completionTime, timecpy, pos, taskI, self.energy, startedCpy, newcapacityRem, currList, currPen, newBuff, usedEnergy)
                            else:
                                # print(str(i)+"start", time, nodes[i].startTime, timecpy)
                                    self.tempSTN(nodes, completionTime, timecpy, pos, taskI, enRem, startedCpy, newcapacityRem, currList, currPen, newBuff, usedEnergy)
                        currList.pop()

    def getSTN(self, task):
            # print(self.attributee, task.type)
        if(self.attribute[int(task.type)]=="1"):

            self.retList = list()
            self.minPen = 10e9
            self.minTT = 10e9
            self.minCT = 10e9
            self.minUsedEnergy = 0
            temp  = self.tasks.copy()
            temp.append(task)
            nodes = {}
            currNodes = set()
            for i in temp:
                nodes[i.taskID] = i
                currNodes.add(i.taskID)
            # generateSTN(0,startPos, currNodes, energy, set(), capacity)
            currList = []
            energyRem = self.energy
            print()
            self.tempSTN(nodes,0,  0, self.currPos, currNodes, energyRem, set(), self.capacity, currList, 0, self.BUFFER, 0)
            # print(self.retList)
            return self.minPen, self.minCT,self.minTT, self.eff, self.minUsedEnergy, energyRem
        else:
            return 10e9,10e9, 10e9, self.eff, 10e9
    
    def addTask(self, task):
        self.retList = list()
        self.minPen = 10e9
        self.minTT = 10e9
        self.minCT = 10e9
        self.minUsedEnergy = 0
        self.tasks.append(task)
        nodes = {}
        currNodes = set()
        for i in self.tasks:
            nodes[i.taskID] = i
            currNodes.add(i.taskID)

        # generateSTN(0,startPos, currNodes, energy, set(), capacity)
        currList = []
        energyRem = self.energy
        self.tempSTN(nodes, 0, 0,self.currPos, currNodes, energyRem, set(), self.capacity, currList, 0, self.BUFFER, 0)
        self.finalList = self.retList.copy()
        self.finalPen = self.minPen
        self.finalTT = self.minTT
        self.finalCT = self.minCT
        self.finalUsedEnergy = self.minUsedEnergy
        # self.finalEnergy = self.energy - energyRem
    def __init__(self, robotID) -> None:
        self.robotID = robotID
        with open('/Users/tejas_sriganesh/Desktop/graph.yaml') as f:
            props = yaml.load(f, Loader=yaml.SafeLoader)
            # print(data)
        # print(props["td"][self.robotID[:2]]["capacity"])
        props = props["td"][self.robotID[:2]]
        self.capacity = props["capacity"]
        self.mass = props["mass"]
        self.velocity = props["velocity"]
        self.energy = props["energy"]
        self.currPos = props["start"]
        self.currCarr = 0
        self.BUFFER = props["BUFFER"]
        self.attribute  = props["attribute"]
        self.eff=0
        for i in props["attribute"]:
            if i =="1":
                self.eff+=1
        self.eff = self.eff/len(props["attribute"])
        self.retList = list()
        self.minPen = 10e9
        self.minTT = 10e9
        self.minCT = 10e9
        self.minUsedEnergy = 0
        self.tasks = list()
        self.finalList = list()
        self.finalPen = 0
        self.finalTT = 0
        self.finalCT = 0
        self.finalUsedEnergy = 0
x=1
total_penalty_initial=0
total_penalty_final=0
pens=[]
prepaths=[]
prepathpens=[]
postpaths=[]
postpathpens=[]
tasks_accepted_for_penalty_lowering=0
tasks_accepted_for_saving_efficient_bots=0
total_task_count=0
toten=0
ct=0
for x in range (1):
    filep = open("trial.txt", "a")
    TASKCOUNT = 50
    total_task_count=TASKCOUNT
    R1COUNT = 20
    R2COUNT = 20
    R3COUNT = 20
    R4COUNT = 20
    roboList = list()
    SOTArobotList1 = list()
    SOTArobotList2 = list()
    SOTArobotList3 = list()
        # print(SOTArobotList3)
        
    for i in range(R1COUNT):
        roboList.append(Robot("r1"+str(i)))
        SOTArobotList1.append(Robot("r1"+str(i)))
        SOTArobotList2.append(Robot("r1"+str(i)))
        SOTArobotList3.append(Robot("r1"+str(i)))
    # print(SOTArobotList3)

    for i in range(R2COUNT):
        roboList.append(Robot("r2"+str(i)))
        SOTArobotList1.append(Robot("r2"+str(i)))
        SOTArobotList2.append(Robot("r2"+str(i)))
        SOTArobotList3.append(Robot("r2"+str(i)))

    for i in range(R3COUNT):
        roboList.append(Robot("r3"+str(i)))
        SOTArobotList1.append(Robot("r3"+str(i)))
        SOTArobotList2.append(Robot("r3"+str(i)))
        SOTArobotList3.append(Robot("r3"+str(i)))

    for i in range(R4COUNT):
        roboList.append(Robot("r4"+str(i)))
        SOTArobotList1.append(Robot("r4"+str(i)))
        SOTArobotList2.append(Robot("r4"+str(i)))
        SOTArobotList3.append(Robot("r4"+str(i)))

    # print("asd")
    # for i in roboList:
    #     print(i.robotID)


    # %%
    ## Demo Tasks
    MIN_VEL = 0.5
    taskQ = []
    for i in range(TASKCOUNT):
        task = TD_task()
        task.taskID= i
        task.arrivalTime = randint(0,10)
        task.demand = randint(10, 40)
        task.destination = choice(list(points.keys()))
        task.pickup = choice(list(points.keys()))
        while task.pickup in ["dock_1", "dock_2", "dock_3"]:
            task.pickup = choice(list(points.keys()))
        while task.destination in ["dock_1", "dock_2", "dock_3", task.pickup]:
            task.destination = choice(list(points.keys()))
        task.startTime = task.arrivalTime+randint(0,10)
        deadline = task.arrivalTime+int(dist[places[task.pickup]][places[task.destination]]/MIN_VEL)
        # print(deadline,'deadline')
        # task.finishTime = deadline
        task.finishTime = randint(deadline, 10*deadline)
        # print(task.finishTime,'finish_time')
        # print(random_number)
        #task.finishTime  = randint(0,1.25*TASKCOUNT)
        task.timeconstraint = randint(0,1)  # 0 for soft, 1 for hard
        task.type = randint(0,6)
        taskQ.append(task)        
    # for i in taskQ:
    #     print(i)
    #     print()
    # %%
    totalRejected = 0
    AcceptedWP= 0 #Accepted request without any penalty
    AcceptedP = 0 #Accepted request with penalty
    for i in taskQ:
        bestRobot = "asd"
        bestBid = [10e9,10e9,10e9, 10e9]
        # print(roboList)
        for j in roboList:
            if(j.attribute[int(i.type)]=="1"):
                bid = j.getSTN(i)
                # print(i.taskID,j.robotID, i.type, i.demand, i.startTime, j.capacity)
                # print(bid)
                if (bestBid[0]>bid[0])or (bestBid[0]==bid[0] and bestBid[3]>bid[3]) or (bestBid[0]==bid[0] and bestBid[3] == bid[3] and bestBid[2]>bid[2] and bestBid[2]==bid[2] and bestBid[5]>bid[5]):
                        bestRobot = j
                        bestBid = bid
        #print(bestRobot.robotID)
        if(bestBid[0]>=10e9):
            # print("rej", bestBid[0])
            totalRejected+=1
        else:
            # print(bestBid[0])
            if(bestBid[0] ==0):
                AcceptedWP+=1
            else:
                AcceptedP+=1
            
            bestRobot.addTask(i)

    

    SOTAtotalRejected1 = 0
    SOTAAcceptedWP1= 0
    SOTAAcceptedP1 = 0
    for i in taskQ:
                SOTAbestRobot1 = "asd"
                SOTAbestBid1 = [10e9,10e9,10e9, 10e9]
                # print(roboList)
                for j in SOTArobotList1:
                    if(j.attribute[i.type]== '1'):
                        SOTABid1 = j.getSTN(i)
                        # print(i.taskID,j.robotID,i.type)
                        # print(SOTABid1)
                        if (SOTAbestBid1[1]>SOTABid1[1]):
                            SOTAbestRobot1 = j
                            SOTAbestBid1 = SOTABid1
                    
            #  print(bestRobot.robotID)
                if(SOTAbestBid1[0]>=10e9):
                    SOTAtotalRejected1+=1
                else:    
                    if( SOTAbestBid1[0] ==0):
                        SOTAAcceptedWP1+=1
                    else:
                        SOTAAcceptedP1+=1                    
                    SOTAbestRobot1.addTask(i)

    SOTAtotalRejected2 = 0
    SOTAAcceptedWP2= 0
    SOTAAcceptedP2 = 0
    for i in taskQ:
                SOTAbestRobot2 = "asd"
                SOTAbestBid2 = [10e9,10e9,10e9,10e9]
                # print(roboList)
                for j in SOTArobotList2:
                    if(j.attribute[i.type]== '1'):
                        SOTABid2 = j.getSTN(i)
                        # print(i.taskID,j.robotID,i.type)
                        # print(SOTABid2)
                        if (SOTAbestBid2[2]>SOTABid2[2] ):
                                SOTAbestRobot2 = j
                                SOTAbestBid2 = SOTABid2
                            
                        #  print(bestRobot.robotID)
                if(SOTAbestBid2[0]>=10e9):
                    SOTAtotalRejected2+=1
                else:
                        if( SOTAbestBid2[0] ==0):
                            SOTAAcceptedWP2+=1
                        else:
                            SOTAAcceptedP2+=1  
                        SOTAbestRobot2.addTask(i)
    
    # SOTAtotalRejected3 = 0 
    # SOTAAcceptedWP3= 0
    # SOTAAcceptedWP3 = 0     
    # SOTAAcceptedP3 = 0         
    # for i in taskQ:
    #             SOTAbestRobot3 = "asd"
    #             SOTAbestBid3 = [10e9,10e9,10e9, 10e9]
    #             # print(roboList)
    #             for j in SOTArobotList3:
    #                 if(j.attribute[i.type]== '1'):
    #                     SOTABid3 = j.getSTN(i)
    #                     # print(i.taskID,j.robotID,i.type)
    #                     # print(SOTABid1)
    #                     if (SOTAbestBid3[1]>SOTABid3[1]):
    #                         SOTAbestRobot3 = j
    #                         SOTAbestBid3 = SOTABid3
                    
    #         #  print(bestRobot.robotID)
    #             if(SOTAbestBid3[0]>=10e9):
    #                 SOTAtotalRejected3+=1
    #             else:
    #                 if( SOTAbestBid3[0] ==0):
    #                      SOTAAcceptedWP3+=1
    #                 else:
    #                     SOTAAcceptedP3+=1  
    #                 SOTAbestRobot3.addTask(i)

                    


    totalPen = 0
    maxTT = 0
    totalCT = 0
    totalEnergyUsed = 0
    totalTT = 0
    for i in roboList:
            # print(i.robotID, len(i.finalList), i.finalPen)
        totalPen+=i.finalPen
        totalTT+=i.finalTT
        totalCT+=i.finalCT
        totalEnergyUsed+= i.finalUsedEnergy
        maxTT = max(i.finalTT, maxTT)
        # print("Our Algo")
        # print(totalPen, maxTT, totalRejected) 

    SOTAtotalPen1 = 0
    SOTAtotalCT1 = 0
    SOTAtotalTT1 = 0
    SOTAmaxTT1 = 0
    SOTAtotalEnergyUsed1 = 0
    for i in SOTArobotList1:
        # print(i.robotID, len(i.finalList), i.finalPen)
        SOTAtotalPen1+=i.finalPen
        SOTAtotalTT1+=i.finalTT
        SOTAtotalCT1 += i.finalCT
        SOTAtotalEnergyUsed1+= i.finalUsedEnergy
        SOTAmaxTT1 = max (i.finalTT, SOTAmaxTT1)

    SOTAtotalPen2 = 0
    SOTAtotalTT2 = 0
    SOTAmaxTT2 = 0
    SOTAtotalCT2 = 0
    SOTAtotalEnergyUsed2 = 0
    for i in SOTArobotList2:
        # print(i.robotID, len(i.finalList), i.finalPen)
        SOTAtotalPen2+=i.finalPen
        SOTAtotalTT2+=i.finalTT
        SOTAtotalCT2+=i.finalCT
        SOTAtotalEnergyUsed2+= i.finalUsedEnergy
        SOTAmaxTT2 = max (i.finalTT, SOTAmaxTT2)

    # SOTAtotalPen3 = 0
    # SOTAmaxTT3 = 0
    # SOTALtotalCT3 = 0
    # SOTAtotalEnergyUsed3 = 0
    # SOTAtotalTT3 = 0
    # for i in SOTArobotList3:
    #         # print(i.robotID, len(i.finalList), i.finalPen)
    #     SOTAtotalPen3+=i.finalPen
    #     SOTAtotalTT3+=i.finalTT
    #     SOTALtotalCT3+=i.finalCT
    #     SOTAtotalEnergyUsed3+= i.finalUsedEnergy
    #     SOTAmaxTT3 = max(i.finalTT, SOTAmaxTT3)
        # print("Our Algo")
        # print(totalPen, maxTT, totalRejected) 
    robots_mit_task =[]
    print("stn calc")
    
    for i in roboList:
       
       if len(i.tasks)!=0:
        robots_mit_task.append(i)
    #print(robots_mit_task)
    minimum_penalty=10000
    minimum_penalty_robot=0
    my=[] 
    tasdic={}
    flg=3
    for i in robots_mit_task:
     for j in i.tasks:
      penalty=i.getSTN(j)[0]
         #if(penalty>0):
     total_penalty_initial+=penalty
    total_penalty_final=total_penalty_initial
    for i in robots_mit_task:
     minimum_penalty_robot=i
     delta_penalty=0
     minimum_penalty=10000
     compt=0
     for j in i.tasks:
        firstpath=[]
        fppen=0
        finpath=[]
        finpen=0
        x1=[]
        x2=[]
        start_endpoints_for_task=[[points[j.pickup]['x'],points[j.pickup]['y']],[points[j.destination]['x'],points[j.destination]['y']]]
        for jj in i.finalList:
            x1.append([points[jj[0]]['x'],points[jj[0]]['y']])
        x1.insert(0,[points[i.currPos]['x'],points[i.currPos]['y']])
        #print(x1)
        path_pretransfer=ffp(x1)
        r=tuple(path_pretransfer[0])
        intr=0
        path_pretransfer.pop(0)     
        path_pretransfer.insert(0,r)  
        for ii in robots_mit_task:
         dem=0
         for u in ii.tasks:
             dem+=u.demand
         caprem=ii.capacity-dem
         if(ii.attribute[j.type]=='1' and caprem>j.demand):
          if(i.robotID!=ii.robotID):
            for jj in ii.finalList:
             x2.append([points[jj[0]]['x'],points[jj[0]]['y']])
            x2.insert(0,[points[ii.currPos]['x'],points[ii.currPos]['y']])
            dista=100000
            clp=0
            if(start_endpoints_for_task[1] in x2):
                print("")
            else:
                for o in x2:
                    d=math.sqrt((o[0]-start_endpoints_for_task[1][0])**2+(o[1]-start_endpoints_for_task[1][1])**2)
                    if d <dista:
                        dista=d
                        clp=x2.index(o)
                x2.insert(clp+1,start_endpoints_for_task[1])
            print(x2,x1,i.robotID,ii.robotID)
            final_path_post_transfer=ffp(x2)
            r=tuple(final_path_post_transfer[0])
            final_path_post_transfer.pop(0)     
            final_path_post_transfer.insert(0,r)
            endpoint_index_in_initial_path_pretransfer=path_pretransfer.index(tuple(start_endpoints_for_task[1]))
            startpoint_index_in_initial_path_pretransfer=path_pretransfer.index(tuple(start_endpoints_for_task[0]))
            endpoint_index_in_final_path_post_transfer=final_path_post_transfer.index(tuple(start_endpoints_for_task[1]))  
            op1=path_pretransfer[startpoint_index_in_initial_path_pretransfer:endpoint_index_in_initial_path_pretransfer+1]
            op2=final_path_post_transfer[:endpoint_index_in_final_path_post_transfer+1]

            intersection1=list(set(op1) & set(op2))
            if(len(intersection1)==0):
                continue
            else:
             intersection=intersection1[0]
            intersectionpoint_index_in_initial_path_pretransfer=path_pretransfer.index(intersection)
            intersectionpoint_index_in_final_path_post_transfer=final_path_post_transfer.index(intersection)
            time=j.finishTime-j.startTime
            completion_time_pretransfer=endpoint_index_in_initial_path_pretransfer/i.velocity
            completion_time_posttransfer=intersectionpoint_index_in_initial_path_pretransfer/i.velocity+(endpoint_index_in_final_path_post_transfer-intersectionpoint_index_in_final_path_post_transfer)/ii.velocity
            penalty_pretransfer=max(completion_time_pretransfer-time,0)
            penalty_posttransfer=max(completion_time_posttransfer-time,0)
            #pens.append([penalty_pretransfer,penalty_posttransfer,minimum_penalty,penalty_posttransfer<penalty_pretransfer,penalty_posttransfer<minimum_penalty])
            
            if(penalty_posttransfer<penalty_pretransfer):
                if(penalty_posttransfer<minimum_penalty):
                    minimum_penalty=penalty_posttransfer
                    minimum_penalty_robot=ii
                    delta_penalty=penalty_pretransfer-penalty_posttransfer
                    flg=1
                    firstpath=path_pretransfer[:endpoint_index_in_initial_path_pretransfer+1]
                    fppen=penalty_pretransfer
                    finpen=penalty_posttransfer
                    finpath+=final_path_post_transfer[:endpoint_index_in_final_path_post_transfer]
                    
                    
                    
            elif(penalty_posttransfer==penalty_pretransfer and penalty_posttransfer<minimum_penalty):
                if(ii.eff<i.eff):
                    minimum_penalty=penalty_posttransfer
                    minimum_penalty_robot=ii
                    delta_penalty=penalty_pretransfer-penalty_posttransfer
                    flg=2
                    #firstpath=path_pretransfer[:endpoint_index_in_initial_path_pretransfer+1]
                    #finpath+=path_pretransfer[:intersectionpoint_index_in_initial_path_pretransfer]+final_path_post_transfer[intersectionpoint_index_in_final_path_post_transfer:endpoint_index_in_final_path_post_transfer]
                    
        total_penalty_final-=delta_penalty
        if(flg==1):
         r="by lower penalty"
         tasks_accepted_for_penalty_lowering+=1
         print(i.robotID,"to",ii.robotID,r)
         prepaths.append(firstpath)
         prepathpens.append(fppen)
         postpaths.append(finpath)
         postpathpens.append(finpen)
        elif(flg==2):
         r="by lower efficiency"
         tasks_accepted_for_saving_efficient_bots+=1
         print(i.robotID,"to",ii.robotID,r)
         #prepaths.append(firstpath)
         #postpaths.append(finpath)
        
    w.append(r)
    print("\n\n\n\n\n\n")
    # print("For SOTA1")
    # print(TASKCOUNT, SOTAtotalPen1, SOTAtotalTT1, SOTAmaxTT1, SOTAtotalRejected1, SOTAtotalCT1, SOTAtotalEnergyUsed1, SOTAAcceptedWP1, SOTAAcceptedP1)
    # print("For SOTA2")
    # print(TASKCOUNT, SOTAtotalPen2, SOTAtotalTT2, SOTAmaxTT2, SOTAtotalRejected2, SOTAtotalCT2, SOTAtotalEnergyUsed2, SOTAAcceptedWP2, SOTAAcceptedP2) 
    #"|fin sum penalty",total_penalty_final,"|tasks accepted for transfer=",accep,"|tasks rejected for transfer=",total_task_count*1-accep,"|energy used=",toten)
    w=0
    # for i in robots_mit_task:
    #     print(i.finalList)
    # for i in pens:
    #     print(i[0],i[1])
    # for i in SOTArobotList:
    #     print(i.finalPen, i.finalTT)
    # print("For SOTA3")
    # print(TASKCOUNT,SOTAtotalPen3, SOTAtotalTT3, SOTAmaxTT3, SOTAtotalRejected3, SOTALtotalCT3, SOTAtotalEnergyUsed3, SOTAAcceptedWP3, SOTAAcceptedP3) 
    print("Ashish sirs algorithm")
    print(TASKCOUNT,totalPen, totalTT, maxTT, totalRejected, totalCT, totalEnergyUsed,AcceptedWP, AcceptedP) 
    print("for transfer")
    print("initial penalt without transfers=",total_penalty_initial,"|final penalty with transfer",total_penalty_final,"|tasks transfered for penalty",tasks_accepted_for_penalty_lowering,"|tasks transferred for efficiency",tasks_accepted_for_saving_efficient_bots,"|tasks rejeted",total_task_count*(1+x)-tasks_accepted_for_saving_efficient_bots-tasks_accepted_for_penalty_lowering)
    
    filep.write(str(TASKCOUNT)+','+str(SOTAtotalPen1)+',' +str(SOTAtotalTT1)+ ',' +str(SOTAmaxTT1)+ ',' +str(SOTAtotalRejected1)+',' + str(SOTAtotalCT1) + ',' + str(SOTAtotalEnergyUsed1) + ',' +str(SOTAAcceptedWP1) +',' +str( SOTAAcceptedP1) + ','
                +str(SOTAtotalPen2) + ',' + str(SOTAtotalTT2) + ',' + str(SOTAmaxTT2) + ',' + str(SOTAtotalRejected2) +',' +str(SOTAtotalCT2) + ',' + str(SOTAtotalEnergyUsed2) + ','  +str(SOTAAcceptedWP2) +',' +str( SOTAAcceptedP2) + ','
                +str(totalPen) + ',' + str(totalTT) + ',' + str(maxTT) + ',' + str(totalRejected) +','+ str(totalCT) + ',' + str(totalEnergyUsed) + ','+ str(AcceptedWP) +',' +str( AcceptedP) + ',''\n')
    
#print("init sum penalty=",total_penalty_initial)
#"|fin sum penalty",total_penalty_final,"|tasks accepted for transfer=",accep,"|tasks rejected for transfer=",total_task_count*1-accep,"|energy used=",toten)
#print(pens)
with open('prepaths.dat', 'wb') as file:
    pickle.dump(prepaths, file)
with open('postpaths.dat', 'wb') as file:
    pickle.dump(postpaths, file)
if(len(prepaths)>0):
    print("initial penalty:",prepathpens[0])
    print("final penalty",postpathpens[0])

filep.close()
