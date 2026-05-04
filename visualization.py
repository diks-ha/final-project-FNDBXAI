import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import re
import os

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('data/dataset.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_text(text):
    """Basic text preprocessing for visualization"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def create_figures_directory():
    """Create figures directory if it doesn't exist"""
    if not os.path.exists('figures'):
        os.makedirs('figures')

def plot_class_distribution(df):
    """Plot class distribution"""
    plt.figure(figsize=(4, 3))

    # Count plot
    ax = sns.countplot(data=df, x='label', palette='Set2')
    plt.title('Distribution of Real vs Fake News', fontsize=16, fontweight='bold')
    plt.xlabel('News Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_text_length_distribution(df):
    """Plot text length distribution by class"""
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Text Analysis: Length and Word Count Distributions', fontsize=16, fontweight='bold')

    # Character length histogram
    sns.histplot(data=df, x='text_length', hue='label', ax=axes[0,0], alpha=0.7, bins=50)
    axes[0,0].set_title('Character Length Distribution')
    axes[0,0].set_xlabel('Character Count')
    axes[0,0].set_ylabel('Frequency')

    # Word count histogram
    sns.histplot(data=df, x='word_count', hue='label', ax=axes[0,1], alpha=0.7, bins=50)
    axes[0,1].set_title('Word Count Distribution')
    axes[0,1].set_xlabel('Word Count')
    axes[0,1].set_ylabel('Frequency')

    # Box plot for character length
    sns.boxplot(data=df, x='label', y='text_length', ax=axes[1,0])
    axes[1,0].set_title('Character Length by News Type')
    axes[1,0].set_xlabel('News Type')
    axes[1,0].set_ylabel('Character Count')

    # Box plot for word count
    sns.boxplot(data=df, x='word_count', ax=axes[1,1])
    axes[1,1].set_title('Word Count by News Type')
    axes[1,1].set_xlabel('News Type')
    axes[1,1].set_ylabel('Word Count')

    plt.tight_layout()
    plt.savefig('figures/text_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\nText Length Statistics:")
    print(df.groupby('label')['text_length'].describe())
    print("\nWord Count Statistics:")
    print(df.groupby('label')['word_count'].describe())

def create_word_clouds(df):
    """Create word clouds for real and fake news"""
    # Separate real and fake news
    real_text = ' '.join(df[df['label'] == 'REAL']['text'].apply(preprocess_text))
    fake_text = ' '.join(df[df['label'] == 'FAKE']['text'].apply(preprocess_text))

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Word Clouds: Real vs Fake News', fontsize=16, fontweight='bold')

    # Real news word cloud
    real_wc = WordCloud(width=800, height=600, background_color='white',
                       max_words=100, colormap='Blues').generate(real_text)
    axes[0].imshow(real_wc, interpolation='bilinear')
    axes[0].set_title('Real News Word Cloud', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Fake news word cloud
    fake_wc = WordCloud(width=800, height=600, background_color='white',
                       max_words=100, colormap='Reds').generate(fake_text)
    axes[1].imshow(fake_wc, interpolation='bilinear')
    axes[1].set_title('Fake News Word Cloud', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('figures/word_clouds.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_words(df, top_n=20):
    """Plot most common words by class"""
    def get_top_words(text_series, n=top_n):
        all_words = []
        for text in text_series:
            words = preprocess_text(text).split()
            all_words.extend(words)

        # Remove common stop words
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                         'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                         'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'])

        filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
        word_counts = Counter(filtered_words)
        return word_counts.most_common(n)

    real_words = get_top_words(df[df['label'] == 'REAL']['text'])
    fake_words = get_top_words(df[df['label'] == 'FAKE']['text'])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Top {top_n} Most Common Words', fontsize=16, fontweight='bold')

    # Real news top words
    real_df = pd.DataFrame(real_words, columns=['word', 'count'])
    sns.barplot(data=real_df, x='count', y='word', ax=axes[0], palette='Blues_r')
    axes[0].set_title('Real News', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Words')

    # Fake news top words
    fake_df = pd.DataFrame(fake_words, columns=['word', 'count'])
    sns.barplot(data=fake_df, x='count', y='word', ax=axes[1], palette='Reds_r')
    axes[1].set_title('Fake News', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('Words')

    plt.tight_layout()
    plt.savefig('figures/top_words.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_text_statistics(df):
    """Plot various text statistics"""
    df['sentence_count'] = df['text'].str.count('[.!?]')
    df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))
    df['unique_words'] = df['text'].apply(lambda x: len(set(str(x).lower().split())))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Text Statistics Analysis', fontsize=16, fontweight='bold')

    # Sentence count
    sns.boxplot(data=df, x='label', y='sentence_count', ax=axes[0,0])
    axes[0,0].set_title('Sentence Count by News Type')
    axes[0,0].set_xlabel('News Type')
    axes[0,0].set_ylabel('Number of Sentences')

    # Average word length
    sns.boxplot(data=df, x='label', y='avg_word_length', ax=axes[0,1])
    axes[0,1].set_title('Average Word Length by News Type')
    axes[0,1].set_xlabel('News Type')
    axes[0,1].set_ylabel('Average Word Length')

    # Unique words
    sns.boxplot(data=df, x='label', y='unique_words', ax=axes[1,0])
    axes[1,0].set_title('Unique Words Count by News Type')
    axes[1,0].set_xlabel('News Type')
    axes[1,0].set_ylabel('Unique Words Count')

    # Correlation heatmap
    numeric_cols = ['text_length', 'word_count', 'sentence_count', 'avg_word_length', 'unique_words']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig('figures/text_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_visualizations():
    """Generate all visualizations"""
    print("Loading data...")
    df = load_data()

    if df is None:
        print("Could not load data. Please ensure data/dataset.csv exists.")
        return

    print("Creating figures directory...")
    create_figures_directory()

    print("Generating visualizations...")

    print("1. Class distribution...")
    plot_class_distribution(df)

    print("2. Text length analysis...")
    plot_text_length_distribution(df)

    print("3. Word clouds...")
    try:
        create_word_clouds(df)
    except ImportError:
        print("WordCloud not available. Skipping word clouds.")

    print("4. Top words analysis...")
    plot_top_words(df)

    print("5. Text statistics...")
    plot_text_statistics(df)

    print("\nAll visualizations saved to figures/ directory!")

def main():
    generate_visualizations()

if __name__ == "__main__":
    main()
