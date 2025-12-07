import numpy as np
from queue import PriorityQueue

class Path_Planner:

    #Given coordinates tests if its in the array bounds
    def in_bounds(self, map, x, y, z):
        X, Y, Z = map.shape
        return 0 <= x < X and 0 <= y < Y and 0 <= z < Z
        
     #Given a start goal and map finds produces a map of the
     #shortest distance while avoiding risky blocks that could contain obsticals   
    def shortest_path(self, start, goal, map):
        avoid_risk = 1
        #increase variable to make the drone try avoid obsticals to a greater extent
        minimize_distance = 1 
        #increase variable to make the drone fly a shorter path
        goal_x, goal_y, goal_z = goal
        distanceArr = np.full(map.shape, np.inf, dtype=float)
        visited = np.full(map.shape, False)
        visitOrder = PriorityQueue()
        previousNode = np.empty(map.shape, dtype=object)
        x_len, y_len, z_len = map.shape
        directions = [
        (1,0,0), (-1,0,0),
        (0,1,0), (0,-1,0),
        (0,0,1), (0,0,-1)
        ]
        distanceArr[start] = 0
        visitOrder.put((0,start))
        #until every point in the array is visted find the shortest path to it
        while visitOrder.qsize() != 0:
            current_distance,current = visitOrder.get()
            #if a point has already been visited skip
            if visited[current[0], current[1], current[2]]:
                continue
                
            visited[current[0], current[1], current[2]] = True
            
            if current == goal:
                break
            #given a point for all surrounding points add to the visit queue
            for direction in directions:
                x = current[0] + direction[0]
                y = current[1] + direction[1]
                z = current[2] + direction[2]
                #if the point isnt in bounds continue
                if not self.in_bounds(map ,x , y, z):
                    continue
                #cost of a point is its set avoid_risk multiplied by how confident there is a obstacle^2
                #on top of the current distance and its minimize_distance variable
                scaled_point = ((map[x, y, z] + 1000) / 2000) ** 2
                h = abs(x - goal_x) + abs(y - goal_y) + abs(z - goal_z)
                cost = avoid_risk * scaled_point + minimize_distance + current_distance + h
                #if its the lowest current cost to reach that position it is saved
                if cost < distanceArr[x, y, z]:
                    distanceArr[x, y, z] = cost
                    visitOrder.put((cost, (x, y, z)))
                    
                    previousNode[x, y, z] = current
        return previousNode
        
    #Returns the list of nodes that is the shortest path      
    def get_Path(self, previousNode, goal):
          path = []
          current = tuple(goal)
          
          while current is not None:
              x = int(current[0])
              y = int(current[1])
              z = int(current[2])
              path.append((x,y,z))
              current = previousNode[current]
          #path is reversed so it starts at start instead of goal 
          path.reverse()
          return path