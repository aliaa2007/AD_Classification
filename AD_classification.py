#Date: 20 February 2022
# This code is part of the research article:
# ” Machine learning framework for the prediction of Alzheimer’s disease using gene expression data based on efficient gene selection”
#  by: Aliaa El-Gawady1 , Mohamed A. Makhlouf , BenBella S. Tawfik , and Hamed Nassar 
# The article is under consideration by the Symmetry Journal, of the MDPI publishing house.
# The code takes as input an Alzheimer’s disease (AD) gene expression dataset and a set of classifiers.
# It then identifies a minimal subset of relevant genes and the best classifiers.
# Finally it outputs the diagnosis (AD/normal) for a case or a set of cases never seen by the classifier before. 
# For more info contact first author at: alia_saad@ci.suez.edu.eg


import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import *



############# read data
ds=pd.read_excel('........\\dataset.xlsx')
print(np.shape(ds)) #to know the matrix dimensions


############transpose matrix to represent features in cols and samples in rows
ds_t=ds.transpose()
print(np.shape(ds_t))




#######################
features_name=ds_t.iloc[0,:-1].values
print(features_name.shape)
features_nam= np.array(features_name)



features_gene_Id=ds_t.iloc[1,:-1].values
features_Id=ds_t.iloc[2,:-1].values
features= np.array(features_Id[:], dtype=int)

data_set=ds_t.iloc[2:,:-1].values
print ('dataset shape is ' , data_set.shape)
dataset1=np.array(data_set)


############## 

XX= ds_t.iloc[3:,:-1].values
print ('X shape is ' , XX.shape)

X = np.array(XX)


y =ds_t.iloc[3:,-1].values
print ('Y shape is ' , y.shape)
y = np.array(y[:], dtype=int)


############to remove nan values
ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)

################### Normalization MinMaxscaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

X = scaler.fit_transform(X)
print(X)
print(scaler.data_min_)
print(scaler.data_max_)

########################## select features by Chi2

from sklearn.feature_selection import SelectKBest, chi2

K=700
score_chi2=[]
ID_chi=[]
# names_chi=[]
ch2 = SelectKBest(chi2, k=K)
X1= ch2.fit_transform(X, y)

nn1= ch2.get_support()
F1_len=len(nn1)

for i in range ( F1_len):
        if nn1[i]==True:
            
            # print( features_name[i],":",ch2.scores_[i])
            # names_chi.append(features_name[i])
            score_chi2.append(ch2.scores_[i])
            ID_chi.append(features_Id[i])
    # # print( ch2.get_params())
    # ##--------------------------------------------
ID_chi=[int(i) for i in ID_chi] # to convert list from float to int 
m_ID_chi2 = []
for j in range(len(ID_chi)):
    m_ID_chi2.append([ID_chi[j], score_chi2[j]])
m_ID_chi2.sort(key=lambda x: x[1], reverse=True)


################################################################

########################## select features by  ANOVA

from sklearn.feature_selection import f_classif


K=1000
score_anova=[]
ID_anova=[]
Anova_fs = SelectKBest(f_classif, k=K)
X2= Anova_fs.fit_transform(X, y)
nn2= Anova_fs.get_support()
F2_len=len(nn2)

for i in range ( F2_len):
        if nn2[i]==True:
            
            # print( features_name[i],":",ch2.scores_[i])
            # names_anova.append(features_name[i])
            score_anova.append(Anova_fs.scores_[i])
            ID_anova.append(features_Id[i])
  
ID_anova=[int(i) for i in ID_anova] # to convert list from float to int 
m_ID_anova = []
for j in range(len(ID_anova)):
    m_ID_anova.append([ID_anova[j], score_anova[j]])
m_ID_anova.sort(key=lambda x: x[1], reverse=True)


################################################################
########################## select features by mutual information
from sklearn.feature_selection import mutual_info_classif

K=1700
score_MI=[]
ID_MI=[]
MI_fs = SelectKBest(mutual_info_classif, k=K)
X3= MI_fs.fit_transform(X, y)
nn3= MI_fs.get_support()
F3_len=len(nn3)

for i in range ( F3_len):
        if nn3[i]==True:
            
            # print( features_name[i],":",ch2.scores_[i])
            # names_MI.append(features_name[i])
            ID_MI.append(features_Id[i])
            score_MI.append(MI_fs.scores_[i])
  
    # ##--------------------------------------------
ID_MI=[int(i) for i in ID_MI] # to convert list from float to int 
m_ID_MI = []
for j in range(len(ID_MI)):
    m_ID_MI.append([ID_MI[j], score_MI[j]])
m_ID_MI.sort(key=lambda x: x[1], reverse=True)

##########################################
#######to get each pair intersection            
            
in1=list(set(ID_chi)&set(ID_anova))
in2=list(set(ID_chi)&set(ID_MI))
in3=list(set(ID_anova)&set(ID_MI))
####to get union among 3 pairs
intersection=list(set(in1)|set(in2)|set(in3)) 
####to get intersection among 3 feature selection methods
all_ins=list(set(ID_chi)&set(ID_anova)&set(ID_MI)) 

##################################
###to take only the intersected genes between 3 lists
del_list = np.delete(data_set,all_ins,axis=1) 
del_list1=list(del_list)
dell=list(del_list1[0])
dell=[int(x) for x in dell]
inters_list= np.delete(data_set,dell,axis=1) 
all_inter_list=np.array(inters_list)
intersection_3=all_inter_list[1:,:]

#########################################
    
deleted_list = np.delete(data_set,intersection,axis=1) 
deleted_list1=list(deleted_list)
deleted=list(deleted_list1[0])
deleted=[int(x) for x in deleted]

intersected_list= np.delete(data_set,deleted,axis=1) 
inter_list=np.array(intersected_list)
union_3=inter_list[1:,:]

############to remove nan values
ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(union_3)
new_union= ImputedX.transform(union_3)
#############################
ImputedModuleX = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedXX = ImputedModuleX.fit(intersection_3)
new_intersection= ImputedXX.transform(intersection_3)

# ##################
# d1 = np.delete(data_set,in1,axis=1) 
# de1=list(d1)
# del1=list(de1[0])
# del1=[int(x) for x in del1]
# in1_list= np.delete(data_set,del1,axis=1) 
# int1_list=np.array(in1_list)
# intersect_1=int1_list[1:,:]
# ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
# ImputedX = ImputedModule.fit(intersect_1)
# intersect_1= ImputedX.transform(intersect_1)
# #######################
# d2 = np.delete(data_set,in2,axis=1) 
# de2=list(d2)
# del2=list(de2[0])
# del2=[int(x) for x in del2]
# in2_list= np.delete(data_set,del2,axis=1) 
# int2_list=np.array(in2_list)
# intersect_2=int2_list[1:,:]
# ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
# ImputedX = ImputedModule.fit(intersect_2)
# intersect_2= ImputedX.transform(intersect_2)

# #######################
# d3 = np.delete(data_set,in3,axis=1) 
# de3=list(d3)
# del3=list(de3[0])
# del3=[int(x) for x in del3]
# in3_list= np.delete(data_set,del3,axis=1) 
# int3_list=np.array(in3_list)
# intersect_3=int3_list[1:,:]
# ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
# ImputedX = ImputedModule.fit(intersect_3)
# intersect_3= ImputedX.transform(intersect_3)


################### Normalization MinMaxscaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

new_union= scaler.fit_transform(new_union)
print(new_union)
print(scaler.data_min_)
print(scaler.data_max_)


scalerX = MinMaxScaler(feature_range = (0,1))

new_intersection= scalerX.fit_transform(new_intersection)
print(new_intersection)
print(scalerX.data_min_)
print(scalerX.data_max_)
#########################################
########### to split data in to train test split

from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts= train_test_split(new_intersection, y, test_size=1057, random_state=44, shuffle =True)

print('X_train shape is ' , X_tr.shape)
print('X_test shape is ' , X_ts.shape)
print('y_train shape is ' , y_tr.shape)
print('y_test shape is ' , y_ts.shape)

######## split train data using RepeatedStratifiedKFold

from sklearn.model_selection import RepeatedStratifiedKFold
k=10
n=30
Id_case=[]

# acc_arr_A8=[]
# acc_arr_S8=[]
# acc_arr_R8=[]
# acc_arr_L8=[]
rskf = RepeatedStratifiedKFold(n_splits=k,n_repeats=n,random_state=44)
rskf.get_n_splits(X_tr, y_tr)

precision_av=0
Recall_av=0
specificity_av=0
acc_av=0
f1_av=0
roc_auc_av=0
cm_av=0
kappa_av=0

    
for train_index1, test_index1 in rskf.split(X_tr, y_tr):
        X_train1, X_test1 =X_tr[train_index1],X_tr[test_index1]
        y_train1, y_test1 = y_tr[train_index1],y_tr[test_index1]
        #######################################
        
         print('X_train1' , X_train1.shape)
         print(X_train1[0])
         case=X_train1[0]
         Id_case.append(case)
#        print('X_test1 ' , X_test1.shape)
#        print('y_train1' ,y_train1.shape)
#        print('y_test1' , y_test1.shape)
         
        ############ Random forest classifier
        # n_trees =100
        # rf = RandomForestClassifier(n_estimators=n_trees)
        # t0 = time.time()
        # rf.fit(X_train1, y_train1)
        # fit_time = time.time() - t0
        # # print("RF complexity and bandwidth selected and model fitted in %.3f s"
        #   # % fit_time)
        
        # train_score= rf.score(X_train1, y_train1)
        # # print('rf Train Score is : ' , train_score)
        
        # test_score= rf.score(X_test1, y_test1)
        # # print('rf Test Score is : ' ,test_score)
       
        # y_pred = rf.predict(X_test1)
        
        ########### Support Vector Machine classifier
        SVM = SVC(kernel="linear")
        t0 = time.time()
        SVM.fit(X_train1, y_train1)
        fit_time = time.time()- t0
#        print("SVM complexity and bandwidth selected and model fitted in %.3f s"
#        % fit_time)
        
        train_score= SVM.score(X_train1, y_train1)
#        print('SVM Train Score is : ' ,train_score )
        
        test_score= SVM.score(X_test1, y_test1)
#        print('SVM Test Score is : ' , test_score )
        
        y_pred = SVM.predict(X_test1)

           
        
          ############### Logistic Regression classifier
        # LR= LogisticRegression(penalty='l2',solver='sag',random_state=33)
        # t0 = time.time()
        # LR.fit(X_train1, y_train1)
        # fit_time = time.time() - t0
        # #    print("LR complexity and bandwidth selected and model fitted in %.3f s" % fit_time)
        
        # train_score= LR.score(X_train1, y_train1)
        # #    print('LR Train Score is : ' , train_score )
        
        # test_score=LR.score(X_test1, y_test1)
        # #    print('LR Test Score is : ' ,test_score )
       
        # y_pred = LR.predict(X_test1)
        
            
           
        #################### AdaBoost classifier
        # Adab=AdaBoostClassifier(n_estimators=100)
        # t0 = time.time()
        # Adab.fit(X_train1, y_train1)
        # fit_time = time.time() - t0
        # #    print(" Adab complexity and bandwidth selected and model fitted in %.3f s"
        # #      % fit_time)
        
        # train_score=  Adab.score(X_train1, y_train1)
        # #    print(' Adab Train Score is : ' , train_score)
        
        # test_score=  Adab.score(X_test1, y_test1)
        # #    print(' Adab Test Score is : ' ,test_score)
    
        # y_pred =  Adab.predict(X_test1)
        
                
        ####### metrics to evaluate classifier
        
        ############ confusion matrix
        
        cm = confusion_matrix(y_test1, y_pred) 
        # print('Confusion Matrix : \n', cm)
        cm_av+=cm
        
        ################# Accuracy
        accuracy_score=metrics.accuracy_score(y_test1, y_pred)
        acc_arr_S8.append(accuracy_score)
        # print("accuracy_score=",accuracy_score)
        acc_av+=accuracy_score
        
        ################################# f1_score
        from sklearn.metrics import f1_score
        f1_sco=f1_score(y_test1, y_pred)
        # print('f1_score=',f1_sco)        
        f1_av+=f1_sco
        
        ########################### roc_auc_score  
        roc_auc = metrics.roc_auc_score(y_test1,y_pred)
        # print("roc_auc=",roc_auc)
        roc_auc_av+=roc_auc
        #    print("\n")
            
        ################ precision
        PrecisionScore = precision_score(y_test1, y_pred)
        # print(" Precision:",PrecisionScore)
        precision_av+=PrecisionScore
        
        ################Recall       
        Rec=recall_score(y_test1, y_pred)
        # print("recall:", Rec)
        Recall_av+=Rec
        
        ###################specificity
        spec = cm[1,1]/(cm[1,1]+cm[1,0])
        # print('Specificity : ', spec)
        specificity_av+=spec
        
        ############# kappa
        kappa=cohen_kappa_score(y_test1, y_pred)
        # print('kappa:', kappa)
        kappa_av+=kappa
        

# acc_arr_A8
# acc_arr_S8
# acc_arr_R8
# acc_arr_L8      
# print(acc_arr)   
# len(acc_arr)   

print(acc_arr_S8)
print(len(acc_arr_S8))

  
######### get the average of the evaluation metrics            

precision_average=(precision_av/n)/k
Recall_average=(Recall_av/n)/k
spec_average=(specificity_av/n)/k
kappa_average=(kappa_av/n)/k
acc_average= (acc_av/n)/k
f1_average=(f1_av/n)/k
roc_auc_average=(roc_auc_av/n)/k
cm_average=(cm_av/n)



#print("acc_average=",acc_average)
#print("f1_average=",f1_average)
#print("roc_auc_av=",roc_auc_average)
#print("time_average=",time_average)
#print("train_score_av=",train_score_av)
#print("test_score_av=",test_score_av)

print(precision_average)
print(Recall_average)
print(spec_average)
print(kappa_average)
print(acc_average)
print(f1_average)
print(roc_auc_average)
print(cm_average)
print(cm_average.astype(int))



############ SVM applied on the final test set 

test_score= SVM.score(X_ts, y_ts)
print('SVM Test Score is : ' , test_score )
y_pred2 = SVM.predict(X_ts)

###### metrics to evaluate test set 
accuracy_test=metrics.accuracy_score(y_ts, y_pred2)
f1_sco_test=f1_score(y_ts, y_pred2)
roc_auc_test = metrics.roc_auc_score(y_ts, y_pred2)
cm_test = confusion_matrix(y_ts, y_pred2) 
Precision_test = precision_score(y_ts, y_pred2)       
Recall_test=recall_score(y_ts, y_pred2)
spec_test = cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])    
kappa_test=cohen_kappa_score(y_ts, y_pred2)
 
print(Precision_test)        
print(Recall_test) 
print(spec_test)             
print(kappa_test)             
print(accuracy_test)
print(f1_sco_test)
print(roc_auc_test)
print( cm_test)


      