from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MaxAbsScaler
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__, static_url_path='/static')

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfv = pickle.load(file)

tfv_matrix = scipy.sparse.load_npz('tfidf_matrix.npz')

df = pd.read_csv('modified_amazon3.csv')

def plot_bar_with_dashed_line(df, new_product_df, column, y_label):
    plt.figure(figsize=(16, 8), dpi=200)
        
    ax = sns.barplot(data=df, y=column, x='product_name', ci=None, palette='viridis', edgecolor=".2", linewidth=1.5, saturation=0.8, capsize=0.1, errwidth=0.5, dodge=True, width=0.8)
    
    plt.axhline(y=new_product_df[column][0], color='black', linestyle='--', label=new_product_df['product_name'][0])
    
    plt.ylabel(y_label)
    plt.xlabel('Product Name')
    
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor")
    
    plt.tight_layout()

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    img_data = base64.b64encode(img_buf.read()).decode('utf-8')

    plt.close()

    return img_data

def give_rec(new_product_df):
    
    tfv_matrix_new = tfv.transform(new_product_df['about_product'])
    
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix_new)
            
    sig_scores = list(enumerate(sig))
    
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    
    sig_scores = sig_scores[1:71]
    
    product_indices = [i[0] for i in sig_scores]
    
    dfw = df[['product_name','discounted_price','rating', 'rating_count']].iloc[product_indices]
    
    v=dfw['rating_count']
    R=dfw['rating']
    C=dfw['rating'].mean()
    m=dfw['rating_count'].quantile(0.70)
        
    dfw['weighted_average']=((R*v)+ (C*m))/(v+m)
    
    dfw=dfw.sort_values('weighted_average',ascending=False)
    
    dfw['pd'] = abs(dfw['discounted_price'] - new_product_df['discounted_price'][0])  
    
    scaler = MaxAbsScaler()
    dfw['scaled_weighted'] = abs(np.diff(scaler.fit_transform(dfw[['weighted_average', 'pd']]), axis=1))

    dfw = dfw.sort_values('scaled_weighted',ascending=False)
    
    return dfw

def give_rec_w_sentiment(new_product_df):
    
    tfv_matrix_new = tfv.transform(new_product_df['about_product'])
    
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix_new)
            
    sig_scores = list(enumerate(sig))
    
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    
    sig_scores = sig_scores[1:71]
    
    product_indices = [i[0] for i in sig_scores]
    
    dfw = df[['product_name','discounted_price','rating', 'rating_count', 'review_content', 'overall_sentiment_score', 'overall_sentiment_intensity']].iloc[product_indices]
    
    v=dfw['rating_count']
    R=dfw['rating']
    C=dfw['rating'].mean()
    m=dfw['rating_count'].quantile(0.70)
        
    dfw['weighted_average']=((R*v)+ (C*m))/(v+m)
        
    dfw['pd'] = abs(dfw['discounted_price'] - new_product_df['discounted_price'][0])  
    
    scaler = MaxAbsScaler()
    dfw['scaled_weighted'] = abs(np.diff(scaler.fit_transform(dfw[['weighted_average', 'pd']]), axis=1))
    dfw['total_weighted'] = dfw['scaled_weighted'] + dfw['overall_sentiment_intensity']
    dfw = dfw.sort_values('total_weighted',ascending=False)
    
    return dfw

@app.route('/')
def Home():
    return render_template('index2.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        name = request.form['pn']
        about = request.form['ap']
        price = float(request.form['dp'])
        rating = float(request.form['rate'])
        rating_count = float(request.form['rateC'])
        sentiment_analysis = int(request.form['Sentiment Analysis'])
        new_product_df = pd.DataFrame({
            'product_name': [name],
            'about_product': [about],
            'discounted_price': [price],
            'rating': [rating],
            'rating_count': [rating_count]
        })
        if sentiment_analysis == 1:
            fdf = give_rec(new_product_df)
        elif sentiment_analysis == 0:
            fdf = give_rec_w_sentiment(new_product_df)

        # Generate bar plot images for the top 5 products
        bar_plot_rating = None
        bar_plot_price = None
        bar_plot_rating_count = None

        if not fdf.empty:
            bar_plot_rating = plot_bar_with_dashed_line(fdf, new_product_df, 'rating', 'Rating')
            bar_plot_price = plot_bar_with_dashed_line(fdf, new_product_df, 'discounted_price', 'Discounted Price')
            
            bar_plot_rating_count = plot_bar_with_dashed_line(fdf, new_product_df, 'rating_count', 'Rating Count')

        return render_template(
            'recommendations2.html',
            fdf=fdf,
            bar_plot_rating=bar_plot_rating,
            bar_plot_price=bar_plot_price,
            bar_plot_rating_count=bar_plot_rating_count
        )

    else:
        return render_template('index2.html')

if __name__ == "__main__":
    app.run(debug=False)