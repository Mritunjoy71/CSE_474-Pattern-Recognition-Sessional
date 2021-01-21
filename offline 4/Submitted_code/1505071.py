import numpy as np
from scipy.stats import multivariate_normal

def loadfile(filename):
    file = open(filename, "r")
    rows = list()
    for line in file:
        vals = line.split()
        rows.append(vals)
    return rows

class Channel:
    def __init__(self,p):
        input=loadfile('input.txt')
        self.h = np.array(input[0],dtype=float)
        #print(self.h)
        self.n=len(self.h)
        #print(self.n)
        self.mu=float(input[1][0])
        #print(self.mu)
        self.var=np.array(input[1][1],dtype=float)
        #print(self.var)
        self.l=p
        self.omega_dict=None
        self.cluster_means=None
        self.cluster_covs=None
        self.cluster_prior_prob=None


    def train(self,data_train):
        X =[]
        for i in range(len(data_train)-1):
            X_k=float(data_train[i+1])*self.h[0]+float(data_train[i])*self.h[1]\
            +np.random.normal(self.mu,self.var)
            #print(ans)
            X.append(X_k)

        #print('length of clusterwise sample X:',len(X))
        l=self.l
        omega_dict=[]
        cluster_means=[]
        cluster_covs=[]
        cluster_prior_prob=[]
        for i in range(np.power(l+1,2)-1):
            omega_dict.append([])
            cluster_means.append([])
            cluster_covs.append([])
            cluster_prior_prob.append([])

        for i in range(l,len(data_train)):
            omega_bin_str=''
            #here omega_bin_str is 3 bit I_k,I_k-2 ,I_k-1
            for j in range(0,l+1):
                omega_bin_str+=data_train[i-j]
                #print(omega_bin_str)
            #bs=bs[::-1]
            omega_class=int(omega_bin_str,2)
            #print(omega_class)
            X_vec=[]
            for k in range(0,l):
                X_vec.append(X[i-l+k])
            X_vec.reverse()
            omega_dict[omega_class].append(X_vec) 
        #print('length of omega dictionary',len(omega_dict))    
        for i in range(len(omega_dict)):
            omega_mean=np.mean(np.array(omega_dict[i]).T,axis =1)
            omega_cov=np.cov(np.array(omega_dict[i]).T)
            cluster_means[i]=omega_mean
            cluster_covs[i]=omega_cov
            cluster_prior_prob[i]=(len(omega_dict[i]) / (len(X)-1))
            #print("cluster no,cluster mean,cluster cov,cluster prior prob",i,cluster_means[i],cluster_covs[i],cluster_prior_prob[i])

        self.omega_dict=omega_dict
        self.cluster_means=cluster_means
        self.cluster_covs=cluster_covs
        self.cluster_prior_prob=cluster_prior_prob


    def distort_Output(self,data):
        X_list =[]
        for i in range(len(data)-1):
            x_k=float(data[i+1])*self.h[0]+float(data[i])*self.h[1]\
            +np.random.normal(self.mu,self.var)
            X_list.append(x_k)
        return X_list



def binary_conversion(out_int):
    out_str=[]
    for i in range(len(out_int)):
        if(i==0):
            bin_str='{0:03b}'.format(out_int[i])
            out_str.append(bin_str[2])
            out_str.append(bin_str[1])
            out_str.append(bin_str[0])
            #print(bin_str)
        else:
            bin_str='{0:03b}'.format(out_int[i])
            #print(bin_str)
            out_str.append(bin_str[0])
    return out_str


def path_detection(D_arr):
    back_node=0
    out_int=[]
    for i in range(len(D_arr)-1,0,-1):
        #print(i)
        if(i==len(D_arr)-1):
            cluster_row=D_arr[i]
            back_node=np.argmax(cluster_row)
            out_int.append(back_node)
        cluster=(back_node%4)*2
        if(D_arr[i-1][cluster] > D_arr[i-1][cluster+1]):
            back_node=cluster
        else:
            back_node=cluster+1
        out_int.append(back_node)
        #print("back node",back_node)

    out_int.reverse()
    return out_int



def write_out_file(out_str,test_data,file_name):
    accuracy=0
    File=open(file_name, 'w')
    for i in range(len(out_str)):
        if test_data[i]==out_str[i]:
            accuracy+=1
        File.write("%s" % out_str[i])

    print("Accuracy of ",file_name,accuracy*100/len(test_data))


def distance_calculation(Eq_model,test_distort,method):
    if method==1:
        D_arr=np.zeros((len(test_distort)-1,np.power(Eq_model.l+1,2)-1), dtype=float)+np.finfo(np.float).eps
    if method==2:
        D_arr=np.zeros((len(test_distort)-1,np.power(Eq_model.l+1,2)-1), dtype=float)
        #print(D_arr)

    #print(D_arr)
    #print("shape of distance array",D_arr.shape)

    for i in range(len(test_distort)-1):
        if(i==0):
            X_k=[]
            X_k.append(test_distort[i+1])
            X_k.append(test_distort[i])
            for j in range(np.power(Eq_model.l+1,2)-1):
                if method==1:
                    D_arr[i][j]+=np.log(Eq_model.cluster_prior_prob[j])+\
                    multivariate_normal.pdf(X_k, Eq_model.cluster_means[j], Eq_model.cluster_covs[j])
                    #print(multivariate_normal.pdf(X_k, Eq_model.clusterMeans[j], Eq_model.clusterCovs[j]))
                if method==2:  
                    D_arr[i][j]=-1*np.linalg.norm(X_k -Eq_model.cluster_means[j] )
  
        else:
            X_k=[]
            X_k.append(test_distort[i+1])
            X_k.append(test_distort[i])
            for j in range(np.power(Eq_model.l+1,2)-1):
                cluster=(j%4)*2
                if method==1:
                    cluster_max = max(D_arr[i-1][cluster],D_arr[i-1][cluster+1])
                    D_arr[i][j] += cluster_max+np.log(0.5)+\
                    multivariate_normal.pdf(X_k, Eq_model.cluster_means[j], Eq_model.cluster_covs[j])
                    #print(multivariate_normal.pdf(X_k, Eq_model.cluster_means[j], Eq_model.cluster_covs[j]),D_arr[i][j]) 
                if method==2:
                    cluster_min=min(D_arr[i-1][cluster],D_arr[i-1][cluster+1])
                    D_arr[i][j]=cluster_min + (-1)*np.linalg.norm(X_k -Eq_model.cluster_means[j] )
                    #print(cluster_min ,np.linalg.norm(xv -Eq_model.cluster_means[j]),D_arr[i][j])


    return D_arr


def main():
    File=open('train.txt')
    train_data=File.read()
    #print('length of train data:',len(train_data))

    File=open('test.txt')
    test_data=File.read()
    #print('length of test data:',len(test_data))

    l=2
    Eq_model=Channel(l)
    Eq_model.train(train_data)
    #print(Eq_model.cluster_covs)

    test_distort=Eq_model.distort_Output(test_data)
    #print(test_distort)
    #print("length of distorted test vector",len(test_distort))

    D_arr=distance_calculation(Eq_model,test_distort,1)
    out_int=path_detection(D_arr)
    out_str=binary_conversion(out_int)    
    write_out_file(out_str,test_data,'out1.txt') 

    D_arr=distance_calculation(Eq_model,test_distort,2)
    out_int=path_detection(D_arr)
    out_str=binary_conversion(out_int)    
    write_out_file(out_str,test_data,'out2.txt')    
    


if __name__=='__main__':
    main()


