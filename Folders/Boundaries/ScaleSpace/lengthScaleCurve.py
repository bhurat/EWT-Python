def lengthScaleCurve(plane):
    [w,num_iter] = plane.shape
    num_curves = np.sum(plane[:,0])
    lengths = np.ones(num_curves.astype(int))
    indices = np.zeros(num_curves.astype(int))
    current_curve = 0;
 
    for i in range(0,w):
        if plane[i,0] == 1:
            indices[current_curve] = i
            i0 = i
            height = 2
            stop = 0
            while stop == 0:
                flag = 0
                for p in range(-1,2):
                    if (i+p  < 0) | (i + p >= w):
                        continue
                    #If minimum at next iteration of scale-space, increment length
                    #height, minimum location
                    if plane[i + p,height] == 1: 
                        lengths[current_curve] += 1 
                        height += 1
                        i0 += p
                        flag = 1
                        #Stop if pas number of iterations
                        if height >= num_iter:  
                            stop = 1
                        break
                #Stop if no minimum found
                if flag == 0:
                    stop = 1
            #Go to next curve/minimum after done
            current_curve += 1  

    return [lengths, indices]