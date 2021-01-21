import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

count_twod=0
count_ex=0

def Exhaustive(ref,template):
    M = ref.shape[0]
    N = ref.shape[1]
    I=template.shape[0]
    J=template.shape[1]
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    
    cross_cor_coeff = np.zeros((I - M + 1, J - N + 1))
    for i in range(I - M + 1):
        for j in range(J - N + 1):
            cross_cor_coeff[i,j] = np.sum(ref.astype(int) * template[i:i+M, j:j+N].astype(int))/\
                        (np.linalg.norm(ref) * np.linalg.norm(template[i:i+M, j:j+N]))            

         
    #print(template, ref)

    max_val=np.argmax(cross_cor_coeff, axis=None)
    shape=cross_cor_coeff.shape
    index =  np.unravel_index(max_val, shape)
    c=index[0]*shape[0]+index[1]
    global count_ex
    count_ex+=c
    top_left_x,top_left_Y = int(index[0] + M/2), int(index[1] + N/2)

    return top_left_x,top_left_Y




def two_D_logarithmic_search( ref,frame):
    I=frame.shape[0]
    J=frame.shape[1]
    M=ref.shape[0]
    N=ref.shape[1]
    frame_col_size=J
    dist = int(frame_col_size/4)
    center_x , center_y = int(I/2), int(J/2)
    
    correlation_max = -math.inf
    argbest=0,0
    while(True):
        for i in range(-1, 2):
            for j in range(-1, 2):
                #print(center_x+i*dist, center_y+j*dist)
                new_center_x = center_x + i * dist - int(M / 2)
                new_center_y = center_y + j * dist - int(N / 2)
                if new_center_x >= 0 and new_center_y >= 0 and new_center_x + M <= I\
                    and new_center_y + N <= J:
                    #print(new_center_x,new_center_y)
                    corr = np.sum(ref.astype(int) * frame[new_center_x:new_center_x+M, new_center_y:new_center_y+N].astype(int))/\
                        (np.linalg.norm(ref) * np.linalg.norm(frame[new_center_x:new_center_x+M, new_center_y:new_center_y+N]))
                    if corr > correlation_max:
                        correlation_max = corr
                        argbest = i, j
        center_x, center_y = center_x + argbest[0] * dist, center_y + argbest[1] * dist
        dist = int(dist / 2)
    
        if dist < 1: break
        global count_twod
        count_twod+=1    
    top_left_x,top_left_Y = center_x + argbest[0] * dist * 2, center_y + argbest[1] * dist * 2

    return top_left_x,top_left_Y




    

def returnTopLeft(ref,frame, x, y, p, method):
    threshold = lambda x : 0 if x < 0 else x
    xt, yt = method(ref,frame[threshold(x - p): threshold(x + p), threshold(y - p): threshold(y + p)])
    return threshold(x - p) + xt, threshold(y - p) + yt



def video_to_frames(filename): 
    frames = [] # rgb frames 
    cap = cv2.VideoCapture(filename)
    
    while(cap.isOpened()):
        ret, frame = cap.read() 
        if frame is None:
            break
        #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    return np.array(frames)



def task_one():
    frames = video_to_frames('input.mov')
    ref = cv2.imread('reference.jpg')

    cap = cv2.VideoCapture('input.mov')
    out1 = cv2.VideoWriter('exhaustive.mov', cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS),\
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    out2 = cv2.VideoWriter('twoD.mov', cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS),\
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    

    cap.release()
    #print(frames.shape)
    first_frame = frames[0]
    #print(first_frame.shape)
    x, y = Exhaustive(ref,first_frame)
    P=60
    for i in range(1,len(frames)):
        frame=frames[i]
    
        #exhaustive_search
        x, y = returnTopLeft(ref,frame, x, y, P, Exhaustive)
        out_frame = cv2.rectangle(frame,(int(y  - ref.shape[1]/2), int(x - ref.shape[0]/2)), \
                        (int(y + ref.shape[1]/2), int(x + ref.shape[0]/2)), (0, 0, 255), 3)
        
        out1.write(out_frame)
        #two_D_logarithmic_search
        x, y = returnTopLeft(ref,frame, x, y, P, two_D_logarithmic_search)
        out_frame = cv2.rectangle(frame,(int(y  - ref.shape[1]/2), int(x - ref.shape[0]/2)), \
                        (int(y + ref.shape[1]/2), int(x + ref.shape[0]/2)), (0, 0, 255), 3)
        out2.write(out_frame)

    out1.release()
    out2.release()
    #print('average search count of reference by ex:',count_ex/len(frames))
    print('completed')




def task_two():
    frames = video_to_frames('input.mov')
    ref = cv2.imread('reference.jpg')

    #print(frames.shape)
    first_frame = frames[0]

    #print(first_frame.shape)

    x, y = Exhaustive(ref,first_frame)
    P=50
    ctd=[]
    cex=[]
    n=5
    for pp in range(0,n):
        P=P+(pp+1)*10
        global count_twod
        global count_ex
        count_twod=0
        count_ex=0
        for i in range(1,len(frames)):
            frame=frames[i]
            #exhaustive_search
            x, y = returnTopLeft(ref,frame, x, y, P, Exhaustive)
            #two_D_logarithmic_search
            x, y = returnTopLeft(ref,frame, x, y, P, two_D_logarithmic_search)

        ctd.append(count_twod)
        cex.append(count_ex)

    print('Exhaustive search\ttwo D logarithmic search ')
    for i in range(n):
        print(cex[i]/len(frames),'\t',ctd[i]/len(frames))

    #print('average search count of reference by twoD:',count_twod/len(frames))

    print('completed')


def main():
    #task_one()
    task_two()

if __name__ == "__main__":
    main()