# Contents of ~/my_app/streamlit_app.py
import streamlit as st
from PIL import Image, ImageOps
import ydata_profiling   
import numpy as np
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg , predict_model as predict_model_reg, plot_model as plot_model_reg,create_model as create_model_reg
from pycaret.classification import setup, compare_models, blend_models, finalize_model, predict_model, plot_model,create_model
from pycaret.classification import *
#from pycaret.regression import *
from pycaret import *
import tensorflow as tf
import keras.preprocessing.image
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from keras.models import Model
import os;
import seaborn as sns
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import os
from tensorflow.keras.preprocessing import image
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
import streamlit as st

url = "https://www.linkedin.com/in/sokhna-faty-bousso-110891190/"
@st.cache_data
def load_data(file):
    data=pd.read_csv(file)
    return data
     
def identification():
    st.sidebar.markdown('<h1 style="text-align: center;">Identification du polluants 🎈</h1>', unsafe_allow_html=True)
    def main():
        st.markdown('<h1 style="text-align: center;">Identification du polluants</h1>', unsafe_allow_html=True)
        st.markdown('<h1 style="text-align: center;">Base de donnée</h1>',unsafe_allow_html=True)
        col3,col4=st.sidebar.columns(2)
        col3.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
        col4.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
        #st.sidebar.write("<p style='text-align: center;'> Sokhna Faty Bousso : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'>Apprentissage par régression ou classification.</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>Nous allons procéder comme suit :</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>1 - Chargement des données</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>2 - Analyse exploratoire des données</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>3 - Sélection de la cible et de la méthode d'apprentissage</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>4 - Construction du modèle</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>5 - Téléchargement du modèle</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>6 - Prédiction</p>", unsafe_allow_html=True)

        file = st.file_uploader("entrer les données ", type=['csv'])
        if file is not None:
            df=load_data(file)
            df=df[df.columns[1:]]
            #type=st.selectbox("selectionner le target",["Homogene","Heterogene"])
            n=len(df.columns)
            X=df[df.columns[:(n-1)]]# on prend les variables numériques 
            y=df[df.columns[-1]] # le target
            st.dataframe(df.head())
            pr = df.profile_report()
            if st.button('statistique descriptive'):
                 st_profile_report(pr)
            if st.button('Save'):
                 df.to_csv('data.csv')
            target=st.selectbox("selectionner le target",df.columns)
            methode=st.selectbox("selectionner la méthode ",["Regression","Classification"])
            df=df.dropna(subset=target)
            if methode=="Classification":
                if st.button(" les performances du modèle "):
                     setup_data = setup(data=df,target = target,
                        train_size =0.75,categorical_features =None,
                        normalize = False,normalize_method = None,fold=5)
                     r=compare_models(round=2)
                     save_model(r,"best_model")
                     st.success("youpiiiii classification fonctionne \U0001F604")
                     st.write("Performances du modèle :")
                
                     final_model1 = create_model(r,fold=5,round=2)
                     col5,col6=st.columns(2)
                     col5.write('AUC')
                     plot_model(final_model1,plot='auc',save=True)
                     col5.image("AUC.png")
                     col6.write("class_report")
                     plot_model(final_model1,plot='class_report',save=True)
                     col6.image("Class Report.png")
                  
                     col7,col8=st.columns(2)
                     col7.write("confusion_matrix")
                     plot_model(final_model1,plot='confusion_matrix',save=True)
                     col7.image("Confusion Matrix.png")
                     tuned_model = tune_model(final_model1,optimize='AUC',round=2,n_iter=10);# optimiser le modéle
                     col8.write("boundary")
                     plot_model(final_model1 , plot='boundary',save=True)
                     col8.image("Decision Boundary.png")
                    
                     col9,col10=st.columns(2)
                     col9.write("feature")
                     plot_model(estimator = tuned_model, plot = 'feature',save=True)
                     col9.image("Feature Importance.png")
                     col10.write("learning")
                     plot_model(estimator = final_model1, plot = 'learning',save=True)
                     col10.image("Learning Curve.png")
                     with open("best_model.pkl",'rb') as f :
                          st.download_button("Telecharger le pipline du modele" , f, file_name="best_model.pkl")
          
          
            if methode=="Regression":
                if st.button("les performances du modèle "):
                    setup_data = setup_reg(data=df,target = target,
                        train_size =0.75,categorical_features =None,
                        normalize = False,normalize_method = None)
                    r=compare_models_reg()
                    save_model(r,"best_model")
                    st.success("youpiiiii classition fonctionne")
                    final_model1 = create_model_reg(r)
        
        else:
            st.image("https://ilm.univ-lyon1.fr//images/slides/SLIDER10.png")

        
    if __name__ == "__main__":
         main()
    st.markdown('<h1 style="text-align: center;">Prédiction</h1>', unsafe_allow_html=True)

    def main():
        file_to_predict = st.file_uploader("Choisir un fichier à prédire", type=['csv'])
        if file_to_predict is not None:
            #rain(emoji="🎈",font_size=54,falling_speed=5,animation_length="infinite",)
            df_to_predict = load_data(file_to_predict)
            st.subheader("Résultats des prédictions")
            def predict_quality(model, df):
                  predictions_data = predict_model(estimator = model, data = df)
                  return predictions_data
    
            model = load_model('best_model')
            pred=predict_quality(model, df_to_predict)
            st.write("A: 0, A+D: 1, A+G: 2, D: 3, D+G: 4, G: 5")
            st.dataframe(pred[pred.columns[-3:]][-20:])
        else:
            st.image("https://ilm.univ-lyon1.fr//images/slides/Germanium%20ILM.jpg")
            

    if __name__ == "__main__":
         main()
dir_path='mysite/base'
fichier=[]
labels=[]
for i in os.listdir(dir_path):
    fichier.append(dir_path+'/'+i)
    labels.append(i)
l=[]
for j in range(len(labels)):
    if (labels[j].find('A+D')!=-1) | (labels[j].find('D+A')!=-1) | (labels[j].find('DA')!=-1 ):
        l.append('A+D')
    elif labels[j].find('EDTA')!=-1:
        l.append('E')
    elif labels[j].find('AMPA')!=-1:
        l.append('A')
    else :
        l.append('D')
labels=l
base=np.transpose(pd.DataFrame([fichier,labels]))

Base=base
base.columns=['image','labels']
train_df, validate_df = train_test_split(base, test_size=0.3)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
image_size = 224
input_shape = (image_size, image_size, 3)
epochs = 30
batch_size = 10
pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

for layer in pre_trained_model.layers[:15]:
    layer.trainable = False
for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
# Flatten la couche de sortie 1 dimension
x = GlobalMaxPooling2D()(last_output)
# # Ajoutez une couche entièrement connectée avec 512 unités cachées et activation ReLU
x = Dense(512, activation='relu')(x)
# ajouter un taux d'abandon 0.5
x = Dropout(0.5)(x)
# il faut donner le nombre de classe et ajouter l'activation sigmoid
x = layers.Dense(4, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image',
    y_col='labels',
    target_size=(image_size, image_size),
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
model.save('model_final2')
def image():
    st.markdown("# image 3D ❄️")
    st.sidebar.markdown('<h1 style="text-align: center;">Identification d\'images ❄️ </h1>', unsafe_allow_html=True)
    col3,col4=st.sidebar.columns(2)
    col3.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
    col4.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
    st.sidebar.write("<p style='text-align: center;'> Sokhna Faty Bousso : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
    st.sidebar.write("<p style='text-align: center;'> modèle pré-entrainé </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>Nous allons procéder comme suit :</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>1 - Aprés avoir la base d'images , nous avons divisé la base de données en ensembles d'entraînement  et de validation </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>2 - créé un modèle de classification basé sur le modèle VGG16 en utilisant Keras</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>3 -  utilisé les 15 premiers couches du modèle </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>4 - ajouté une couche de classification à la sortie du modèle avec une activation sigmoid pour la classification multi-classes.</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>5 - compilé le modèle en utilisant une fonction de perte de binary_crossentropy (car les étiquettes sont encodées en tant que vecteurs binaires) et un optimiseur SGD</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>6 -  créé des générateurs d'images d'apprentissage  et de validation pour alimenter le modèle pendant l'entraînement, en utilisant ImageDataGenerator de Keras pour augmenter les données d'entraînement (rotation, ré-échelle, retournement, etc.).</p>", unsafe_allow_html=True)
   
    st.markdown('<h1 style="text-align: center;">Prédiction image 3D </h1>', unsafe_allow_html=True)
    #model = load_model('model_final2.pkl')
    f=['A','A+D','D','E']
    from PIL import Image
    def preprocess_image(image_path, target_size=(224, 224)):
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = image.resize(target_size)
        return np.array(image) / 255.0

    def predict_class(model, image_path):
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        index = np.argmax(pred)
        f.sort()
        pred_value = f[index]
        return pred_value

    file = st.file_uploader("Entrer l'image", type=["jpg", "png"])
    if file is None:
        st.text("entrer l'image à prédire")
    else:
        label = predict_class(model, file)
        st.image(file, use_column_width=True)
        st.markdown("## Résultats de la prédiction ")
        st.markdown("## Il s'agit du polluant")
        st.write(label)
    st.image("https://ilm.univ-lyon1.fr//images/slides/SLIDER7.png")


def Quantification():
    st.markdown('<h1 style="text-align: center;"> Quantification du polluants 🎉 </h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<h1 style="text-align: center;"> Quantification du polluants 🎉 </h1>', unsafe_allow_html=True)
    def cal_conc2(x,y,z,h,Ca,Cd):
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
    def find_delimiter(filename):
        sniffer = csv.Sniffer()
        with open(filename) as fp:
             delimiter = sniffer.sniff(fp.read(5000)).delimiter
        return delimiter
    def mono_exp(df,VAR):
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
        for  i in range(int(ncol/2)):
            x=df[df.columns[0]]; # temps
            y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
            popt,pcov=curve_fit(f_decay,x,y,bounds=(0, np.inf));
            df1=df1.append({'A'+VAR.split('/')[-1] :popt[0] , 'Tau'+VAR.split('/')[-1] :popt[1]} , ignore_index=True)
        return(df1)   
    def double_exp(df,VAR):
        T1=0.3
        T2=1.3
        #df=pd.read_csv(VAR,sep=delimit)
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
    def tri_exp(df,VAR):
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
                   popt,pcov=curve_fit(f_decay,x,y,bounds=(0,[yo,+np.inf,bound_c,+np.inf]),method='trf') # On utilise une regression non linéaire pour approximer les courbes de durée de vie  
                   #popt correspond aux paramètres a1,b1,c,r de la fonction f_decay de tels sorte que les valeurs de f_decay(x,*popt) soient proches de y (intensités de fluorescence)
                   break;
                except ValueError:
                    bound_c=bound_c-0.05
                    print("Oops")
            df2=df2.append({"préexpo_"+VAR.split('/')[-1]:2*popt[0],"tau_"+VAR.split('/')[-1]:popt[1]} , ignore_index=True);# Pour chaque solution , on ajoute la préexponentielle et la durée de vie tau à la dataframe
    
            ax.plot(x,y,label="Intensité réelle");
            ax.plot(x,f_decay(x,*popt),label="Intensité estimée");
            ax.set_title(" solution "+df.columns[2*i]);
            ax.set_xlabel('Temps(ms)');
            ax.set_ylabel('Intensité(p.d.u)');
            plt.legend();
        plt.show();
    
        return(df2)
    ## regression avec linearregression
    def regression1(result,std,unk,ss,d):
        concentration=pd.DataFrame(columns=['polyfit'])
        col1, col2 ,col3,col4= st.columns(4)
        col=[col1,col2,col3,col4]
        for t in range(len(ss)): 
            fig, ax = plt.subplots()
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
            mymodel = np.poly1d(np.polyfit(x, y, d)) # polynome de degré 1
            x=x.reshape(-1,1);
            y_intercept = mymodel(0)
            R3=r2_score(y, mymodel(x))
            # tracer les courbes de calibérations 
            #print('\n',f"\033[031m {result.columns[2*t+1][2:]} \033[0m",'\n')
            plt.plot(x, mymodel(x),'m',label='np.polyfit : R² = {}'.format(round(R3,2)))
            plt.xlabel('Concentration solution standard(ppm)')
            plt.ylabel('durée de vie(ms)');
            plt.title('Courbe de calibration'+'du polluant '+result.columns[2*t+1][4:])
            plt.legend();
            col[t].pyplot(fig)
            y_intercept = mymodel(0)
            col[t].write("y_intercept")
            col[t].write(y_intercept)
            # Calcul des racines (x_intercept)
            roots = np.roots(mymodel)
            x_intercepts = [root for root in roots if np.isreal(root)]
            x_inter=fsolve(mymodel,0)
            col[t].write("x_intercept")
            col[t].write(x_inter)
            slope=mymodel.coef[0]
            col[t].write("slope")
            col[t].write(slope)
            x_inter=fsolve(mymodel,0)
            Cx=(y_intercept-tau[0])/slope
            concentration=concentration.append({'polyfit':round(x_inter[0],2)},ignore_index=True)
        return(concentration)
    def fun(tau):
        sum_k=1/tau
        kch=-sum_k+sum_k[0]
        return(sum_k,kch)
    def regression2(result,std,unk,ss,sum_kchel):
        col1, col2 ,col3,col4= st.columns(4)
        col=[col1,col2,col3,col4]
        con_poly3=[]
        con2=[]
        for i in range(len(ss)):
           fig, ax = plt.subplots()
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
           #st.write(f"\033[031m {result.columns[2*i+1][4:]} \033[0m")
           plt.scatter(x, kchel)
           plt.plot(x, mymodel(x),'m')
           plt.xlabel('Concentration solution standard(ppm)');
           plt.ylabel('nombre d\'ion chélaté ' );
           plt.title('Courbe de calibration'+'du polluant '+result.columns[2*i+1][4:])
           plt.legend();
           col[i].pyplot(fig)
           col[i].write(r2_score(kchel, mymodel(x)))
           # Calcul de l'ordonnée à l'origine (y_intercept)
           y_intercept = mymodel(0)
           col[i].write("y_intercept")
           col[i].write(y_intercept)
           # Calcul des racines (x_intercept)
           roots = np.roots(mymodel)
           x_intercepts = [root for root in roots if np.isreal(root)]
           x_inter=fsolve(mymodel,0)
           con_poly3.append(x_inter)
           col[i].write("x_intercept")
           col[i].write(x_inter)
           slope=mymodel.coef[0]
           col[i].write("slope")
           col[i].write(slope)
           xinter=y_intercept/slope
           con2.append(xinter)
        return(con_poly3)
    def regression6(result, std, unk, ss, sum_kchel):
        col1, col2 ,col3,col4= st.columns(4)
        col=[col1,col2,col3,col4]
        con_poly3 = []
        con2 = []
        for i in range(len(ss)):
            fig, ax = plt.subplots()
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
            col[i].pyplot(fig)
            col[i].write(log_r2)
            y_intercept = func(0, *log_params)
            col[i].write("y_intercept")
            col[i].write(y_intercept)
            x_inter = np.exp(-2*log_params[2]/log_params[0]) - log_params[1]
            x_inter=np.array([x_inter])
            col[i].write("x_intercept")
            col[i].write(x_inter)
            slope = -log_params[1] *func(x_inter, *log_params)
            #[i].write("slope")
            #col[i].write(slope)
            con_poly3.append(x_inter)
            con2.append(x_inter)
        return con_poly3
    def regression7(result, std, unk, ss, sum_kchel):
        col1, col2 ,col3,col4= st.columns(4)
        col=[col1,col2,col3,col4]
        con_poly3 = []
        con2 = []
        for i in range(len(ss)):
            fig, ax = plt.subplots()
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
                 return a * np.exp(-b * x) + c
            initialParameters = np.array([1.0, 1.0, 1.0])
            log_params, _= curve_fit(func, x, kchel, initialParameters,maxfev=50000)
            log_r2 = r2_score(kchel,func(x, *log_params))
        
            best_model = func(x, *log_params)
            plt.scatter(x, kchel)
            plt.plot(x, best_model, 'm')
            plt.title('Courbe de calibration'+'du polluant '+result.columns[2*i+1][4:])
            col[i].pyplot(fig)
            col[i].write(log_r2)
            y_intercept = func(0, *log_params)
            col[i].write("y_intercept")
            col[i].write(y_intercept)
            x_inter =  -np.log(-log_params[2] / log_params[0]) / log_params[1]
            x_inter=np.array([x_inter])
            col[i].write("x_intercept")
            col[i].write(x_inter)
            slope = -log_params[1] *func(x_inter, *log_params)
            #col[i].write("slope")
            #col[i].write(slope)
            con_poly3.append(x_inter)
            con2.append(x_inter)
        return con_poly3



    
    Taux4 = pd.DataFrame()
    def main():
        col3,col4=st.sidebar.columns(2)
        col3.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
        col4.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
        st.sidebar.write("<p style='text-align: center;'> Sokhna Faty Bousso : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'>  Mélange de polluants  </p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>1 - Méthode mono_exponentielle : </p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>2 - Méthode double_exponentielle :</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>3 - Méthode gaussienne  : </p>", unsafe_allow_html=True) 
        st.sidebar.markdown("<p style='text-align: center;'>Dans chaque méthode nous allons procéder comme suit : Calcul de la concentration dans chaque série  à partir de (durée de vie et nombre d'ion chélaté ) en fonction de la concentration standard dans chaque mélange avec une regression linéaire et non linéaire afin d'utiliser le systéme d'équation (P) pour trouver la concentration de chaque polluant dans le mélange\n </p>", unsafe_allow_html=True)
        st.latex(r'''P = \begin{cases} -C_{HD} = C_D + K_{A-D}C_A &\text{S1 }  \\ - C_{HA}= C_A + K_{D-A}C_D &\text{S2 } \\- C_{DA} = K_{D-A}C_D^0  &\text{S3 }   \\ - C_{AD} = K_{A-D}C_A^0 &\text{S4 }  \end{cases}''')
        uploaded_files = st.file_uploader("Choisir les fichiers csv ", accept_multiple_files=True)
        st.image("https://ilm.univ-lyon1.fr//images/slides/CARROUSSEL-17.png")
        unk = st.number_input("Volume unk")
        ss1 = st.number_input("Solution standard",value=1, step=1, format="%d")
        rev = st.number_input("Volume rev")
        Ca = st.number_input("Concentration initiale de A",value=1, step=1, format="%d")
        Cd = st.number_input("Concentration initiale de D",value=1, step=1, format="%d")
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1]
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2] # 06-06
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1] 
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1.7]
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # Volume standard 07-06 , 12-06
        #std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.5,1] # Volume standard 08-06 
        #std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # 09-06
        std=[0,0,0.025,0.075,0.125,0.2,0.5,0.7,1,1.5,4] # Volume standard 20-06 , 21-06
        # Calculer ss en fonction de ss1
        ss = [ss1] * 4
        #global Taux4
        if uploaded_files is not None:
            col5,col6,col7=st.columns(3)
            Taux4 = pd.DataFrame()
            if col5.button("Méthode mono_exponentielle"):
                st.latex(r''' \fcolorbox{red}{green}{$f_decay(x,a,tau) =  \epsilon + a\exp (\frac{-x}{tau} ) $}''')
                for uploaded_file in uploaded_files:
                    df = pd.read_csv(uploaded_file, delimiter="\t")
                    Q=mono_exp(df,uploaded_file.name)
                    T=pd.concat([Taux4,Q], axis=1)
                    Taux4=T
                Taux=Taux4
                sum_kchel1=pd.DataFrame() # gaussienne
                sum_kchel2=pd.DataFrame()# double exp
                sum_kchel3=pd.DataFrame() # mono exp
                for j in range(4):
                    tt3=Taux4[Taux4.columns[2*j+1]]
                    s_k=pd.DataFrame(fun(tt3))
                    s_k=s_k.T
                    s_k.columns=['sum_k'+Taux.columns[2*j+1].split('_')[-1],'kchel'+Taux.columns[2*j+1].split('_')[-1]]
                    sum_kchel3=pd.concat([sum_kchel3,s_k],axis=1)
                st.markdown('## <h1 style="text-align: center;"> Les valeurs de Taux et la pré_exponentielle</h1>',unsafe_allow_html=True) 
                st.write(Taux4.style.background_gradient(cmap="Greens")) 
                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de durée de vie</h1>',unsafe_allow_html=True) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(Taux4,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Greens"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression6(Taux4,std,unk,ss,Taux4)
                #col1.write(concentration4)
                #polyfit=concentration4[concentration4.columns[0]]
                concen =pd.DataFrame(concentration4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r2=cal_conc(*concentration4,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Greens"))


                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de nombre d\'ion chélaté </h1>',unsafe_allow_html=True) 
                st.write(sum_kchel3.style.background_gradient(cmap="Greens")) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire </h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentrationC4=regression6(Taux4,std,unk,ss,sum_kchel3)
                concen =pd.DataFrame(concentrationC4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r1=cal_conc(*concentrationC4,Ca,Cd)
                col2.dataframe(r1.style.background_gradient(cmap="Greens"))
                    
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(sum_kchel3,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Greens"))

    
            if col6.button("Méthode double_exponentielle"):
                st.latex(r'''\fcolorbox{red}{blue}{$f_decay(x,a1,t1,a2,t2) =  \epsilon + a1\exp (\frac{-x}{t1} ) +a2\exp (\frac{-x}{t2})$}''')
                st.latex(r'''\fcolorbox{red}{blue}{$Aire = a1t1 + a2t2 $}''')
                Taux4 = pd.DataFrame()
                for uploaded_file in uploaded_files:
                       df = pd.read_csv(uploaded_file, delimiter="\t")
                       Q=double_exp(df,uploaded_file.name)
                       T=pd.concat([Taux4,Q], axis=1)
                       Taux4=T
                Taux=Taux4
                sum_kchel1=pd.DataFrame() # gaussienne
                sum_kchel2=pd.DataFrame()# double exp
                sum_kchel3=pd.DataFrame() # mono exp
                for j in range(4):
                    tt3=Taux4[Taux4.columns[2*j+1]]
                    s_k=pd.DataFrame(fun(tt3))
                    s_k=s_k.T
                    s_k.columns=['sum_k'+Taux.columns[2*j+1].split('_')[-1],'kchel'+Taux.columns[2*j+1].split('_')[-1]]
                    sum_kchel3=pd.concat([sum_kchel3,s_k],axis=1)
                st.markdown('## <h1 style="text-align: center;"> Les valeurs de Taux et la pré_exponentielle</h1>',unsafe_allow_html=True)
                st.write(Taux4.style.background_gradient(cmap="Blues")) 
                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de durée de vie</h1>',unsafe_allow_html=True) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(Taux4,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Blues"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression6(Taux4,std,unk,ss,Taux4)
                #col1.write(concentration4)
                #polyfit=concentration4[concentration4.columns[0]]
                concen =pd.DataFrame(concentration4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r2=cal_conc(*concentration4,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Blues"))

                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de nombre d\'ion chélaté </h1>',unsafe_allow_html=True) 
                st.write(sum_kchel3.style.background_gradient(cmap="Blues")) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentrationC4=regression6(Taux4,std,unk,ss,sum_kchel3)
                concen =pd.DataFrame(concentrationC4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r1=cal_conc(*concentrationC4,Ca,Cd)
                col2.dataframe(r1.style.background_gradient(cmap="Blues"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(sum_kchel3,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Blues"))
            if col7.button("Méthode gaussienne "):
                st.latex(r'''\fcolorbox{red}{purple}{$f_decay (x,a1,t1,c) =  \epsilon + a1\exp (\frac{-x}{t1} )  +\frac{a2}{2}\exp (\frac{-x}{t1+1.177c} ) +\frac{a2}{2}\exp (\frac{-x}{t1-1.177c})$}''')
                Taux4 = pd.DataFrame()
                for uploaded_file in uploaded_files:
                       df = pd.read_csv(uploaded_file, delimiter="\t")
                       Q=tri_exp(df,uploaded_file.name)
                       T=pd.concat([Taux4,Q], axis=1)
                       Taux4=T
                Taux=Taux4
                sum_kchel1=pd.DataFrame() # gaussienne
                sum_kchel2=pd.DataFrame()# double exp
                sum_kchel3=pd.DataFrame() # mono exp
                for j in range(4):
                    tt3=Taux4[Taux4.columns[2*j+1]]
                    s_k=pd.DataFrame(fun(tt3))
                    s_k=s_k.T
                    s_k.columns=['sum_k'+Taux.columns[2*j+1].split('_')[-1],'kchel'+Taux.columns[2*j+1].split('_')[-1]]
                    sum_kchel3=pd.concat([sum_kchel3,s_k],axis=1)
                st.markdown('## <h1 style="text-align: center;"> Les valeurs de Taux et la pré_exponentielle</h1>',unsafe_allow_html=True)
                st.write(Taux4.style.background_gradient(cmap="Purples")) 
                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de durée de vie</h1>',unsafe_allow_html=True) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(Taux4,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Purples"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression6(Taux4,std,unk,ss,Taux4)
                #col1.write(concentration4)
                #polyfit=concentration4[concentration4.columns[0]]
                concen =pd.DataFrame(concentration4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r2=cal_conc(*concentration4,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Purples"))

                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de nombre d\'ion chélaté </h1>',unsafe_allow_html=True) 
                st.write(sum_kchel3.style.background_gradient(cmap="Purples")) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentrationC4=regression6(Taux4,std,unk,ss,sum_kchel3)
                concen =pd.DataFrame(concentrationC4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r1=cal_conc(*concentrationC4,Ca,Cd)
                col2.dataframe(r1.style.background_gradient(cmap="Purples"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(sum_kchel3,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Purples"))
                
    if __name__ == "__main__":
         main()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

page_names_to_funcs = {
    "identification": identification,
    "image": image,
    "Quantification": Quantification,
}

selected_page = st.sidebar.selectbox("Selectionner ", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
