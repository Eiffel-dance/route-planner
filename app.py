import heapq
def plan(width,height,blocked,start,goal):
    blocked=set(blocked); q=[(0,0,start)]; came={}; cost={start:0}; serial=0
    def h(p): return abs(p[0]-goal[0])+abs(p[1]-goal[1])
    while q:
        _,_,cur=heapq.heappop(q)
        if cur==goal:
            path=[cur]
            while path[-1] in came: path.append(came[path[-1]])
            path.reverse(); return {"path":path,"cost":cost[cur],"expanded":len(cost)}
        for nxt in ((cur[0]+1,cur[1]),(cur[0]-1,cur[1]),(cur[0],cur[1]+1),(cur[0],cur[1]-1)):
            if not(0<=nxt[0]<width and 0<=nxt[1]<height) or nxt in blocked: continue
            new=cost[cur]+1
            if new<cost.get(nxt,10**9): cost[nxt]=new; came[nxt]=cur; serial+=1; heapq.heappush(q,(new+h(nxt),serial,nxt))
    return {"path":None,"cost":None,"expanded":len(cost)}
