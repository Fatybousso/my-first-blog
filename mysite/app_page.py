import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt;
from scipy.optimize import curve_fit,fsolve
from scipy.signal import savgol_filter
from scipy import signal
import sympy as sp 
from scipy.integrate import quad
import scipy.integrate as spi
from sklearn import preprocessing
from scipy import stats
from sklearn.linear_model import LinearRegression
from tkinter import *
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sympy import symbols
from sympy import cos, exp
from sympy import lambdify
import statsmodels.formula.api as smf
from sympy import *
import csv
from scipy import optimize
from sklearn.metrics import r2_score#pour calculer le coeff R2
from sklearn.linear_model import RANSACRegressor
from colorama import init, Style
from termcolor import colored


# 
# ###  **<center><font>  Fonction qui détermine le délimiter du fichier   </font></center>**

# In[2]:


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


# 
# 
# ###  **<center><font>  Fonction qui calcul les concentrations finale    </font></center>**

#  - $C_{HD}$ :  la concentration obtenue dans la serie 1
#  - $C_{HA}$ :  concentration obtenue dans la serie 2
#  - $C_{DA}$ :  concentration obtenue dans la serie 3
# -  $C_{AD}$ :  concentration obtenue dans la serie 4
#  - $C_D^0$ : concentration initiale du polluant 1 
#  - $C_A^0$ : concentration initiale du polluant 2

# - $ C_{HD} = C_D + K_{A-D}C_A $     =>  Serie 1 : mélange dans standard 1 (D)
# - $ C_{HA}= C_A + K_{D-A}C_D $      =>  Serie 2 : mélange dans standard 2 (A)
# - $ C_{DA} = K_{D-A}C_D^0 $         =>  Serie 3 : polluant 1 dans standard 2 
# - $C_{AD} = K_{A-D}C_A^0$            =>  Serie 4 : polluant 2 dans la standard 1

# In[3]:


def cal_conc1(x,y,z,h,Ca,Cd):
    a=h/Ca
    a1=z/Cd
    C_A=(y-a1*x)/(1-a1*a)
    C_D=(x-a*y)/(1-a1*a)
    conc=pd.DataFrame([C_A,C_D])
    conc.index=['C_A','C_D']
    return(conc) 
def cal_conc(x,y,z,h,Ca,Cd):
        a=-z/Ca # serie
        a1=-h/Cd
        y1=-y
        y3=-x
        C_A=(a*y3-y1)/(a1*a-1)
        C_D=(a1*y1-y3)/(a1*a-1)
        conc=pd.DataFrame([C_A,C_D])
        conc.index=['C_A','C_D']
        return(conc)


# 
# 
# # **<center><font color='blue'> méthode  monoexponentielle </font></center>**

# # f_decay $(x,a,tau)$ = $ \epsilon + a\exp (\frac{-x}{tau} )  $

# In[4]:


def mono_exp(VAR):
    delimit=find_delimiter(VAR)
    df=pd.read_csv(VAR,sep=delimit);
    #-------------Nettoyage du dataframe----------------#
    for i in df.columns:
        if (df[i].isnull()[0]==True):# On elimine les colonnes vides
            del df[i];
    df=df.dropna(axis=0);#On elimine les lignes contenant des na
    df=df[1:];
    df=df.astype(float); # On convertit les valeurs contenues dans les colonnes en float (à la place de string)
    df=df[df[df.columns[0]]>=0.1]
    ncol=(len(df.columns)) # nombre de colonnes
    najout=(ncol/2)-3; # nombre d'ajouts en solution standard
    #---------------------First step----------------------#
    def f_decay(x,a,b,c):
        return(c+a*np.exp(-x/b));
    df1=pd.DataFrame(columns=['A'+VAR.split('/')[-1],'Tau'+VAR.split('/')[-1]]);
    row=int(len(df.columns)/5)
    row2=int(len(df.columns)/2)
    fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 20))
    for ax, i in zip(axs.flat, range(int(ncol/2))):
        x=df[df.columns[0]]; # temps
        y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
        y=y/max(y)
        popt,pcov=curve_fit(f_decay,x,y,bounds=(0, np.inf));
        df1=df1.append({'A'+VAR.split('/')[-1] :popt[0] , 'Tau'+VAR.split('/')[-1] :popt[1]} , ignore_index=True)
        f=f_decay(x,*popt)
        ax.plot(x,y,label="Intensité réelle");
        ax.plot(x,f,label="Intensité estimée");
        ax.set_title(" solution "+df.columns[2*i]);
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('Intensité(p.d.u)');
        plt.legend();
    plt.show();
    return(df1)   
    


# # **<center><font color='blue'> méthode  double exponentielle </font></center>**

# # f_decay $(x,a1,t1,a2,t2)$ = $ \epsilon + a1\exp (\frac{-x}{t1} ) +a2\exp (\frac{-x}{t2})  $
# ## tau = $ \frac{a1t1^2 + a2t2^2}{ a1t1 + a2t2} $

# In[5]:


def double_exp2(VAR,T1,T2):
    delimit=find_delimiter(VAR)
    df=pd.read_csv(VAR,sep=delimit)
    #-------------Nettoyage du dataframe----------------#
    for i in df.columns:
        if (df[i].isnull()[0]==True):# On elimine les colonnes vides
            del df[i];
    df=df.dropna(axis=0);#On elimine les lignes contenant des na
    df=df[1:];
    df=df.astype(float); # On convertit les valeurs contenues dans les colonnes en float (à la place de string)
    df=df[df[df.columns[0]]>=0.1]
    ncol=(len(df.columns)) # nombre de colonnes
    najout=(ncol/2)-3; # nombre d'ajouts en solution standard
    #---------------------First step----------------------#
    def f_decay(x,a1,a2,r):
        return(r+a1*np.exp(-x/T1)+a2*np.exp(-x/T2));
    
    df1=pd.DataFrame(columns=['A_'+VAR.split('/')[-1],'Aire_'+VAR.split('/')[-1]]);
    for i in range(int(ncol/2)):
        x=df[df.columns[0]]; # temps
        y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
        y=list(y)
        y0=max(y)#y[1]
        popt,pcov=curve_fit(f_decay,x,y,bounds=(0.1,[y0,y0,+np.inf]));
        #tau=(popt[0]*(popt[1])**2+popt[2]*(popt[3])**2)/(popt[0]*(popt[1])+popt[2]*(popt[3]))
        A1=popt[0]*T1
        A2=popt[1]*T2
        A=A1+A2 # l'aire sous la courbe de l'intensité de fluorescence 
        df1=df1.append({'A_'+VAR.split('/')[-1] :A1,'Aire_'+VAR.split('/')[-1] :A} , ignore_index=True);
    return(df1)   
    


# 
# 
# # **<center><font color='blue'>   méthode  gaussiennes  </font></center>**

# # f_decay $(x,a1,t1,c)$ = $ \epsilon + a1\exp (\frac{-x}{t1} )  +\frac{a2}{2}\exp (\frac{-x}{t1+1.177c} ) +\frac{a2}{2}\exp (\frac{-x}{t1-1.177c})  $

# In[6]:


def tri_exp(VAR):
    delimit=find_delimiter(VAR)
    df=pd.read_csv(VAR,sep=delimit);
    for i in df.columns:
        if (df[i].isnull()[0]==True): # On elimine les colonnes vides
            del df[i];
    df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
    df=df[1:];
    df=df.astype(float); # On convertit les valeurs contenues dans les colonnes en float (à la place de string)
    df=df[df[df.columns[0]]>=0.1]
    ncol=(len(df.columns)) # nombre de colonnes
    def f_decay(x,a1,b1,c,r): # Il s'agit de l'équation utilisée pour ajuster l'intensité de fluorescence en fonction du temps(c'est à dire la courbe de durée de vie)
        return(a1*np.exp(-x/b1)+(a1/2)*np.exp(-x/(b1+1.177*c))+(a1/2)*np.exp(-x/(b1-1.177*c))+r)
                                           
    df2=pd.DataFrame(columns=["préexpo_"+VAR.split('/')[-1],"tau_"+VAR.split('/')[-1]]); # Il s'agit du dataframe qui sera renvoyé par la fonction
    #### Ajustement des courbes de durée de vie de chaque solution en fonction du temps#### 
    print('polluant '+VAR.split('/')[-1].split('.')[0])
    row=int(len(df.columns)/5)
    row2=int(len(df.columns)/2)
    fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 20))
    for ax, i in zip(axs.flat, range(int(ncol/2))):
        x=df[df.columns[0]]; # temps
        y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
        y=list(y)
        yo=max(y)#y[1]
        bound_c=1
        while True:
            try:
                popt,pcov=curve_fit(f_decay,x,y,bounds=(0,[yo,+np.inf,bound_c,+np.inf]),method='dogbox') # On utilise une regression non linéaire pour approximer les courbes de durée de vie  
                #popt correspond aux paramètres a1,b1,c,r de la fonction f_decay de tels sorte que les valeurs de f_decay(x,*popt) soient proches de y (intensités de fluorescence)
                break;
            except ValueError:
                bound_c=bound_c-0.05
                print("Oops")
        df2=df2.append({"préexpo_"+VAR.split('/')[-1]:2*popt[0],"tau_"+VAR.split('/')[-1]:popt[1]} , ignore_index=True);# Pour chaque solution , on ajoute la préexponentielle et la durée de vie tau à la dataframe
        y=np.log(y)
        f=np.log(f_decay(x,*popt))
        ax.plot(x,y,label="log  Intensité réelle");
        ax.plot(x,f,label="log Intensité estimée");
        ax.set_title(" solution "+df.columns[2*i]);
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('log(Intensité(p.d.u))');
        ax.legend();
    plt.show();
    
    return(df2)


# 
# 
# 
# # **<center><font color='blue'> Fonction pour  regression linéaire </font></center>**

# ## Calcule concentration en fonction de durée de vie 
# Nous avons utilisé trois fonction pour la regression linéaire : 
#   - LinearRegression()
#   - RANSACRegressor()
#   - np.polyfit()

# In[7]:


## regression avec linearregression
def regression1(result,std,unk,ss):
    concentration=pd.DataFrame(columns=['polyfit','stats_lingress','ransac'])
    for t in range(len(ss)): 
        ax1=plt.subplot(211)
        tau=result[result.columns[2*t+1]]
        cc=tau;
        y=np.array(cc); 
        std=np.array(std)
        conc=ss[t]*std/unk
        x=conc;
        n=len(x)
        x=x[1:(n-1)]
        y=y[1:(n-1)]
        plt.scatter(x,y);
        ####Construction de la courbe de calibration des durées de vie 
        # les modéles 
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x, y);# On effectue une régression linéaire entre les concentrations en solution standard (x) et les durées de vie (y)
        modeleReg1=LinearRegression()
        modeleReg2=RANSACRegressor() # regression optimal
        mymodel = np.poly1d(np.polyfit(x, y, 1)) # polynome de degré 1
        x=x.reshape(-1,1);
        modeleReg1.fit(x,y);
        modeleReg2.fit(x,y)
        fitLine1 = modeleReg1.predict(x);# valeurs predites de la regression
        slope2 = modeleReg2.estimator_.coef_[0]
        intercept2 = modeleReg2.estimator_.intercept_
        inlier_mask = modeleReg2.inlier_mask_
        fitLine2 = modeleReg2.predict(x);# valeurs predites de la regression
        y_intercept = mymodel(0)
        R2=modeleReg2.score(x,y)
        R1=modeleReg1.score(x,y)
        r_value = r2_score(y, fitLine2)
        residuals = y - fitLine2
        R3=r2_score(y, mymodel(x))
        # tracer les courbes de calibérations 
        print('\n',f"\033[031m {result.columns[2*t+1][4:]} \033[0m",'\n')
        plt.plot(x, fitLine1, c='r',label='stats.linregress : R² = {} '.format(round(R1,2)));
        plt.plot(x, mymodel(x),'m',label='np.polyfit : R² = {}'.format(round(R3,2)))
        plt.plot(x, fitLine2, color="black",label='RANSACRegressor : R² = {} '.format(round(R2,2)))
        plt.xlabel('Concentration solution standard(ppm)');
        plt.ylabel('durée de vie(ms)');
        plt.title('Courbe de calibration'+'du polluant '+result.columns[2*t+1][4:])
        plt.legend();
        plt.show();
        y_intercept = mymodel(0)
        print("y_intercept:", y_intercept)
        # Calcul des racines (x_intercept)
        roots = np.roots(mymodel)
        x_intercepts = [root for root in roots if np.isreal(root)]
        x_inter=fsolve(mymodel,0)
        print("x_intercepts:", x_inter)
        slope=mymodel.coef[0]
        print("slope", slope)
        # calcul des concentrations
        Cx1=-(intercept1)/slope1;
        Cx2=-(intercept2)/slope2
        std_err = np.std(residuals)
        roots = np.roots(mymodel)
        x_intercepts = [root for root in roots if np.isreal(root)]
        x_inter=fsolve(mymodel,0)
        equation_text1 = 'y = {}x + {}'.format(slope1, intercept1)
        equation_text2 = 'y = {}x + {}'.format(slope2, intercept2)
        print("stats.linregress :",equation_text1, '\n'," polyfit :",equation_text2, '\n', "RANSACReg : " , mymodel)
        concentration=concentration.append({'polyfit':round(x_inter[0],2),'stats_lingress':round(Cx1,2),'ransac':round(Cx2,2)},ignore_index=True)
    return(concentration)


# In[65]:


def regression6(result, std, unk, ss, sum_kchel):
    con_poly3 = []
    con2 = []
    for i in range(len(ss)):
        tau = result[result.columns[2*i+1]]
        cc = tau
        y = np.array(cc)
        std = np.array(std)
        conc = ss[i] * std / unk
        x = conc
        n = len(x)
        x = x[1:(n-1)]
        kchel = sum_kchel[sum_kchel.columns[2*i+1]]
        sum_k = sum_kchel[sum_kchel.columns[2*i+1]]
        kchel = kchel[1:(n-1)]

        def func(x, a, b, c): # x-shifted log
              return a*np.log(x + b)/2+c
        initialParameters = np.array([1.0, 1.0, 1.0])
        log_params, _= curve_fit(func, x, kchel, initialParameters,maxfev=50000)
        log_r2 = r2_score(kchel,func(x, *log_params))
        
        best_model = func(x, *log_params)
        plt.scatter(x, kchel)
        plt.plot(x, best_model, 'm')
        plt.show()

       
        y_intercept = func(0, *log_params)
        print("y_intercept:", y_intercept)
        x_inter = np.exp(-2*log_params[2]/log_params[0]) - log_params[1]
        x_inter=np.array([x_inter])
        print("x_intercepts:", x_inter)
        slope = -log_params[1] *func(x_inter, *log_params)
        con_poly3.append(x_inter)
        con2.append(x_inter)
    return con_poly3



# In[ ]:





# In[ ]:





# In[287]:





# In[ ]:





# In[ ]:





# ## calcul concentration en fonction du nombre d'ion chélaté 

# In[9]:


def regression3(result,std,unk,ss,sum_kchel):
    con_poly3=[]
    con2=[]
    for i in range(len(ss)):
        ax1=plt.subplot(211)
        tau=result[result.columns[2*i+1]]
        cc=tau;
        y=np.array(cc); 
        std=np.array(std) 
        conc=ss[i]*std/unk
        x=conc;
        n=len(x)
        x=x[1:(n-1)]
        kchel=sum_kchel[sum_kchel.columns[2*i+1]]
        sum_k=sum_kchel[sum_kchel.columns[2*i+1]]
        kchel=kchel[1:(n-1)]
        mymodel = np.poly1d(np.polyfit(x, kchel, 1))
        print('\n',f"\033[031m {result.columns[2*i+1][4:]} \033[0m",'\n')
        plt.scatter(x, kchel)
        plt.plot(x, mymodel(x),'m')
        plt.xlabel('Concentration solution standard(ppm)');
        plt.ylabel('nombre d\'ion chélaté ' );
        plt.title('Courbe de calibration'+'du polluant '+result.columns[2*i+1][4:])
        plt.legend();
        plt.show() 
        print(mymodel,'\n','R² = {:.5f}'.format(r2_score(kchel, mymodel(x))))
        # Calcul de l'ordonnée à l'origine (y_intercept)
        y_intercept = mymodel(0)
        print("y_intercept:", y_intercept)
        # Calcul des racines (x_intercept)
        roots = np.roots(mymodel)
        x_intercepts = [root for root in roots if np.isreal(root)]
        x_inter=fsolve(mymodel,0)
        print("x_intercepts:", x_inter)
        con_poly3.append(x_inter)
        slope=mymodel.coef[0]
        xinter=y_intercept/slope
        con2.append(xinter)
    return(con_poly3)


# # **<center><font color='blue'> Fonction pour  regression non linéaire </font></center>**

#  - Polynôme de degré 3 

# In[10]:


def regression2(result,std,unk,ss,sum_kchel):
    con_poly3=[]
    con2=[]
    for i in range(len(ss)):
        ax1=plt.subplot(211)
        tau=result[result.columns[2*i+1]]
        cc=tau;
        y=np.array(cc); 
        std=np.array(std) 
        conc=ss[i]*std/unk
        x=conc;
        n=len(x)
        x=x[1:(n-1)]
        kchel=sum_kchel[sum_kchel.columns[2*i+1]]
        sum_k=sum_kchel[sum_kchel.columns[2*i+1]]
        kchel=kchel[1:(n-1)]
        mymodel = np.poly1d(np.polyfit(x, kchel, 3))
        print('\n',f"\033[031m {result.columns[2*i+1][4:]} \033[0m",'\n')
        plt.scatter(x, kchel)
        plt.plot(x, mymodel(x),'m')
        plt.xlabel('Concentration solution standard(ppm)');
        plt.ylabel('nombre d\'ion chélaté ' );
        plt.title('Courbe de calibration'+'du polluant '+result.columns[2*i+1][4:])
        plt.legend();
        plt.show() 
        print(mymodel,'\n','R² = {:.5f}'.format(r2_score(kchel, mymodel(x))))
        # Calcul de l'ordonnée à l'origine (y_intercept)
        y_intercept = mymodel(0)
        print("y_intercept:", y_intercept)
        # Calcul des racines (x_intercept)
        roots = np.roots(mymodel)
        x_intercepts = [root for root in roots if np.isreal(root)]
        x_inter=fsolve(mymodel,0)
        print("x_intercepts:", x_inter)
        con_poly3.append(x_inter)
        slope=mymodel.coef[0]
        print("slope", slope)
        xinter=y_intercept/slope
        con2.append(xinter)
    return(con_poly3)


# # **<center><font color='blue'>  Séléctionner les 4 Séries  </font></center>**

# In[11]:


def browseFiles2():
	filename = filedialog.askopenfilenames(initialdir = "http://localhost:8888/tree/Stage",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# In[48]:


VARS=browseFiles2()


# 
# # **<center><font color='blue'>  Entrer les valeurs   </font></center>**

# In[49]:


unk=2.8 # volume inconnue
unk=2.6 # le 19-06
unk=2.6
std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2] # 06-06
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1] 
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1.7]
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # Volume standard 07-06 , 12-06
#std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.5,1] # Volume standard 08-06 
#std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # 09-06
std=[0,0,0.025,0.075,0.125,0.2,0.5,0.7,1,1.5,4] # Volume standard 20-06 , 21-06
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,3]  # 15-06  , 16-06 
#std=[0,0,0.025,0.075,0.125,0.2,0.5,0.7,1,1.5,3] 
ss1=100 # solution standard serie 1
ss2=100 # standard serie 2
ss3=100 # standard serie 3
ss4 =100 # standard serie 4
rev=0.4 # volume reveratrice
Ca=10 # concentration initiale du polluant A dans la serie 4
Cd=10 # concentration initiale du polluant D dans la serie 3


# 
# 
# # **<center><font color='blue'>   Resultats    </font></center>**

# 
# 
# 
# ## **<center><font>  mono_exponentielle </font></center>**

# 
# 
# 
# ### **<center><font>    I )  On fit l'intensité avec une mono_exponentielle puis on calcul les concentrations en utilisant une regressions linéaire ensuite non lineaire ( degré 3)    </font></center>**

# In[50]:


Taux4=pd.DataFrame()
for VAR in VARS:
    #print("Serie : " , VAR.split('/')[-1])
    Q=mono_exp(VAR)
    T=pd.concat([Taux4,Q], axis=1)
    Taux4=T
result4=pd.DataFrame()
j=[1,1,2,2,3,3,4,4]
for i in j:
    for k in range(len(Taux4.columns)) : 
        
        if Taux4.columns[k].find('S'+str(i))!=-1:
            result4=pd.concat([result4,Taux4[Taux4.columns[k]]],axis=1)
result4 = result4.loc[:,~result4.columns.duplicated()]       


# ## **<center><font> Tableau qui contient les taux et les pré_exponentielle de chaque échantillon dans chacune des séries   </font></center>**

# In[41]:


result4.style.background_gradient(cmap="Greens")


# 
# ## **<center><font> I-1) Calcul des concentrations en fonction durée de vie par une regression linéaire  </font></center>**

# In[21]:


ss=[ss1,ss2,ss3,ss4]
concentration4=regression1(result4,std,unk,ss) 


# ## **<center><font>  resultats des concentrations obtenuent dans chaque serie </font></center>**

# In[22]:


concentration4
serie=['s1','s2','s3','s4']
concentration4.index=serie
concentration4.style.background_gradient(cmap="Greens")


# ### Les concentrations finales pour chaque polluant 

# In[23]:


polyfit=concentration4[concentration4.columns[0]]
stats_lingress=concentration4[concentration4.columns[1]]
ransac=concentration4[concentration4.columns[2]]
r2=cal_conc(*polyfit,Ca,Cd)
r3=cal_conc(*stats_lingress,Ca,Cd)
r4=cal_conc(*ransac,Ca,Cd)
r5=pd.concat([r2,r3,r4],axis=1)
r5.columns=['polyfit','stats_lingress','ransac']
r5.style.background_gradient(cmap="Greens")


# In[ ]:





# 
# ## **<center><font> I-1) Calcul des concentrations en fonction nombre d'ion chélaté  par une regression linéaire  </font></center>**

# ### Calcul kchel et sum_k pour chaque serie 

# In[18]:


def fun(tau):
    sum_k=1/tau
    kch=-sum_k+sum_k[0]
    return(sum_k,kch)
sum_kchel1=pd.DataFrame() # gaussienne
sum_kchel2=pd.DataFrame()# double exp
sum_kchel3=pd.DataFrame() # mono exp
for j in range(4):
    tt3=result4[result4.columns[2*j+1]]
    s_k=pd.DataFrame(fun(tt3))
    s_k=s_k.T
    s_k.columns=['sum_k'+result4.columns[2*j+1].split('_')[-1],'kchel'+result4.columns[2*j+1].split('_')[-1]]
    sum_kchel3=pd.concat([sum_kchel3,s_k],axis=1)


# ### Tableau qui donne le nombre d'ion chélaté et le pourcentage de chaque taux pour chaque série 

# In[19]:


sum_kchel3.style.background_gradient(cmap="Greens")


# In[26]:


ss=[ss1,ss2,ss3,ss4]
concentrationCC4=regression3(result4,std,unk,ss,sum_kchel3) 


# In[27]:


concentrationCC4
cc=pd.DataFrame(concentrationCC4)
cc.style.background_gradient(cmap="Greens")


# In[28]:


r2=cal_conc(*concentrationCC4,Ca,Cd)
r2.style.background_gradient(cmap="Greens")


# 
# ## **<center><font>   I-2 ) Calcul des concentrations en utilisant une regression non lineaire ( degré 3) </font></center>**

# In[66]:


ss=[ss1,ss2,ss3,ss4]
c=regression6(result4,std,unk,ss,sum_kchel3) 


# In[76]:


r2=cal_conc(*c,Ca,Cd)
r2.style.background_gradient(cmap="Greens")


# In[29]:


ss=[ss1,ss2,ss3,ss4]
concentrationC4=regression2(result4,std,unk,ss,sum_kchel3)


# ## Resultast des concentrations obtenuent dasn chaque serie 

# In[93]:


concen =pd.DataFrame(concentrationC4)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Greens")


# ### Resultats des concentrations de chaque polluant

# In[94]:


r1=cal_conc(*concentrationC4,Ca,Cd)
r1.style.background_gradient(cmap="Greens")


# In[ ]:





# 
# # **<center><font>  méthode double_exponentielle   </font></center>**

# In[133]:


Taux2=pd.DataFrame()
for VAR in VARS:
    #print("Serie : " , VAR.split('/')[-1])
    Q=double_exp(VAR)
    T2=pd.concat([Taux2,Q], axis=1)
    Taux2=T2
result2=pd.DataFrame()
j=[1,1,2,2,3,3,4,4]
for i in j:
    for k in range(len(Taux2.columns)) : 
        if Taux2.columns[k].find('S'+str(i))!=-1:
            result2=pd.concat([result2,Taux2[Taux2.columns[k]]],axis=1)
result2 = result2.loc[:,~result2.columns.duplicated()]   


# ## Tableau qui donne les taux et les pré_exponentielle pour chaque série

# In[96]:


result2.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  calcule de la concentration en fonction durée de vie  pour la  regression linéaire  </font></center>**

# In[97]:


ss=[ss1,ss2,ss3,ss4]
concentrationC3=regression1(result2,std,unk,ss) 


# In[98]:


serie=['s1','s2','s3','s4']
concentrationC3.index=serie
concentrationC3.style.background_gradient(cmap="Blues")


# ## Resultats des concentrations pour chaque série 

# In[99]:


polyfit=concentrationC3[concentrationC3.columns[0]]
stats_lingress=concentrationC3[concentrationC3.columns[1]]
ransac=concentrationC3[concentrationC3.columns[2]]
r2=cal_conc(*polyfit,Ca,Cd)
r3=cal_conc(*stats_lingress,Ca,Cd)
r4=cal_conc(*ransac,Ca,Cd)
r5=pd.concat([r2,r3,r4],axis=1)
r5.columns=['polyfit','stats_lingress','ransac']
r5.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  calcule de la concentration en fonction de nombre d'ion chélaté  pour la  regression linéaire  </font></center>**

# In[100]:


for j in range(4):
    tt2=result2[result2.columns[2*j+1]]
    s_k=pd.DataFrame(fun(tt2))
    s_k=s_k.T
    s_k.columns=['sum_k'+result2.columns[2*j+1].split('_')[-1],'kchel'+result2.columns[2*j+1].split('_')[-1]]
    sum_kchel2=pd.concat([sum_kchel2,s_k],axis=1)


# In[101]:


result10=pd.DataFrame()
j=[1,1,2,2,3,3,4,4]
for i in j:
    for k in range(len(sum_kchel2.columns)) : 
        if sum_kchel2.columns[k].find('S'+str(i))!=-1:
            result10=pd.concat([result10,sum_kchel2[sum_kchel2.columns[k]]],axis=1)
result10 = result10.loc[:,~result10.columns.duplicated()]    


# In[102]:


sum_kchel2=result10
result10.style.background_gradient(cmap="Blues")


# In[103]:


ss=[ss1,ss2,ss3,ss4]
concentrationCC3=regression3(result2,std,unk,ss,sum_kchel2) 


# In[104]:


concentrationCC3
cc=pd.DataFrame(concentrationCC3)
cc.style.background_gradient(cmap="Blues")


# In[105]:


r2=cal_conc(*concentrationCC3,Ca,Cd)
r2.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  calcule de la concentration pour la  regression non linéaire  </font></center>**

# In[106]:


ss=[ss1,ss2,ss3,ss4]
concentration3=regression2(result2,std,unk,ss,sum_kchel2) 


# In[107]:


concen =pd.DataFrame(concentration3)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Blues")


# ## Calcul concentration final 

# In[109]:


r1=cal_conc(*concentration3,Ca,Cd)
r1.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  méthode gaussiennes    </font></center>**

# In[143]:


Taux=pd.DataFrame()
for VAR in VARS:
    #print("Serie : " , VAR.split('/')[-1])
    Q=tri_exp(VAR)
    T=pd.concat([Taux,Q], axis=1)
    Taux=T
result=pd.DataFrame()
j=[1,1,2,2,3,3,4,4]
for i in j:
    for k in range(len(Taux.columns)) : 
        if Taux.columns[k].find('S'+str(i))!=-1:
            result=pd.concat([result,Taux[Taux.columns[k]]],axis=1)
result = result.loc[:,~result.columns.duplicated()]       


# 
# # **<center><font>  resultats calcul de Taux et préexponentielle   </font></center>**

# In[111]:


result.style.background_gradient(cmap="Purples") 


# 
# # **<center><font>  resultats de la concentration en fonction durée de vie par  regression  linéaire  </font></center>**

# In[112]:


ss=[ss1,ss2,ss3,ss4]
concentrationC1=regression1(result,std,unk,ss) 


# In[113]:


concentrationC1
serie=['s1','s2','s3','s4']
concentrationC1.index=serie
concentrationC1.style.background_gradient(cmap="Purples")


# ## Calcul concentration final 

# In[114]:


polyfit=concentrationC1[concentrationC1.columns[0]]
stats_lingress=concentrationC1[concentrationC1.columns[1]]
ransac=concentrationC1[concentrationC1.columns[2]]
r2=cal_conc(*polyfit,Ca,Cd)
r3=cal_conc(*stats_lingress,Ca,Cd)
r4=cal_conc(*ransac,Ca,Cd)
r5=pd.concat([r2,r3,r4],axis=1)
r5.columns=['polyfit','stats_lingress','ransac']
r5.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  resultats de la concentration en fonction de nombre d'ion chélaté par  regression  linéaire  </font></center>**

# In[115]:


for j in range(4):
    tt1=result[result.columns[2*j+1]]
    s_k=pd.DataFrame(fun(tt1))
    s_k=s_k.T
    s_k.columns=['sum_k'+result.columns[2*j+1].split('_')[-1],'kchel'+result.columns[2*j+1].split('_')[-1]]
    sum_kchel1=pd.concat([sum_kchel1,s_k],axis=1)


# In[116]:


sum_kchel1.style.background_gradient(cmap="Purples")


# In[117]:


ss=[ss1,ss2,ss3,ss4]
concentrationCC1=regression3(result,std,unk,ss,sum_kchel1) 


# In[118]:


cc=pd.DataFrame(concentrationCC1)
cc.style.background_gradient(cmap="Purples")


# In[119]:


r1=cal_conc(*concentrationCC1,Ca,Cd)
r1.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  resultats regression non  linéaire  </font></center>**

# In[120]:


ss=[ss1,ss2,ss3,ss4]
concentration1=regression2(result,std,unk,ss,sum_kchel1) 


# In[121]:


concen =pd.DataFrame(concentration1)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  Concentration final   </font></center>**

# In[122]:


r2=cal_conc(*concentration1,Ca,Cd)
r2.style.background_gradient(cmap="Purples")


'''
        st.code(code,language="python")

   


page_names_to_funcs = {
    "Quantification": Quantification,
    "code python":code_python 
}

selected_page = st.sidebar.selectbox("Selectionner ", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

