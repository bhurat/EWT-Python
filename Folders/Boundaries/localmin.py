def localmin(f):
    w = len(f)
    minima = np.zeros(w)
    for i in range(0,w):
        minima[i] = 1
        right = 1
        while 1:
            if i - right >= 0:
                if f[i - right] < f[i]:
                    minima[i] = 0
                    break
                elif f[i - right] == f[i]:
                    right += 1
                else:
                    break
            else:
                break
        if minima[i] == 1:
            left = 1
            while 1:
                if i + left < w:
                    if f[i+left] < f[i]:
                        minima[i] = 0
                        break
                    elif f[i+left] == f[i]:
                        left += 1
                    else:
                        break
                else:
                    break
    i = 0
    while i < w:
        if minima[i] == 1:
            j = i
            flat_count = 1
            flag = 0
            while (j+1 < w) and (minima[j+1] == 1):
                minima[j] = 0
                flat_count += 1
                j += 1
                flag = 1
                print(minima)
            if flag == 1:
                minima[j - np.floor(flat_count/2).astype(int)] = 1
                minima[j] = 0
                i = j
        i += 1
                
    return minima