import numpy as np
from queue import PriorityQueue

class Path_Planner:

    def test(self):
        return "hello"
        
    def in_bounds(self, map, x, y, z):
        X, Y, Z = map.shape
        return 0 <= x < X and 0 <= y < Y and 0 <= z < Z
          
    def shortest_path(self, start, goal, map):
        avoid_risk = 1
        minimize_distance = 1 
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
        
        while visitOrder.qsize() != 0:
            current_distance,current = visitOrder.get()
            if visited[current[0], current[1], current[2]]:
                continue
            visited[current[0], current[1], current[2]] = True
            if current == goal:
                break
                
            for direction in directions:
                x = current[0] + direction[0]
                y = current[1] + direction[1]
                z = current[2] + direction[2]
                if not self.in_bounds(map ,x , y, z) or map[x, y, z] <= 0:
                    continue
                cost = avoid_risk * -np.log(map[x,y,z]) + minimize_distance + current_distance 
                if cost < distanceArr[x, y, z]:
                    distanceArr[x, y, z] = cost
                    visitOrder.put((cost, (x, y, z)))
                    previousNode[x, y, z] = current
        return previousNode
            
    def get_Path(self, previousNode, goal):
          path = []
          current = tuple(goal)
          while current is not None:
              path.append(current)
              current = previousNode[current]
          path.reverse()
          return path