# ✈️ Airline Passenger Satisfaction Clustering

This project applies **clustering techniques** to the **Airline Passenger Satisfaction Dataset** to identify distinct passenger segments. Using **K-Means clustering**, exploratory data analysis, and machine learning techniques, we uncover insights to improve customer satisfaction and airline operations.

## 📂 Project Overview
- **Objective**: Identify customer segments based on travel behavior, satisfaction, and service ratings.
- **Techniques Used**:
  - Data Preprocessing (handling missing values, encoding categorical data)
  - Exploratory Data Analysis (EDA) using visualizations
  - **K-Means Clustering** with optimal **K selection** (Elbow Method, Silhouette Score)
  - PCA for dimensionality reduction and cluster visualization
- **Key Findings**:
  - **Six passenger segments** identified, varying in age, travel patterns, loyalty, and satisfaction.
  - Business travelers show high satisfaction despite poor Wi-Fi ratings.
  - Economy passengers report lower satisfaction due to booking experience and onboard service quality.
  - Long delays heavily impact specific traveler groups.

## 🛠️ Tech Stack
- **Programming Language**: Python 🐍
- **Libraries**:
  - `pandas` - Data preprocessing
  - `numpy` - Numerical operations
  - `matplotlib` & `seaborn` - Data visualization
  - `sklearn` - Machine learning (clustering, PCA)
  
## 📊 Data Preprocessing
- **Missing Values**: Filled using the **median** to prevent outlier influence.
- **Categorical Encoding**:
  - **Binary Mapping**: Gender, Customer Type, Travel Type, Satisfaction
  - **One-Hot Encoding**: Travel Class
- **Feature Scaling**: Standardized numeric features using `StandardScaler` to improve clustering performance.

## 🔍 Exploratory Data Analysis (EDA)
- **Histograms** of numerical features (age, flight distance, satisfaction ratings)
- **Bar charts** for categorical data (gender, travel type, loyalty status)
- **Correlation Heatmap**: Highlights strong relationships between departure and arrival delays.
- **Pair Plots**: Visualizes satisfaction trends based on delays and travel type.

## 🤖 Clustering Approach
- **K-Means Clustering**:
  - Optimal K determined using **Elbow Method & Silhouette Score**.
  - PCA applied for visualization of cluster distributions.

## 📌 Key Clusters & Insights
| Cluster | Age Group | Travel Type | Satisfaction | Issues Identified | Suggested Improvements |
|---------|----------|------------|--------------|-------------------|------------------------|
| 0 | 32 | Short-haul | Low | Poor booking experience, low Wi-Fi quality | Improve digital experience, upgrade Wi-Fi |
| 1 | 32 | Short-medium haul | Low | Low service ratings (seat, food, cleanliness) | Enhance onboard service quality |
| 2 | 41 | Business | High | Low Wi-Fi, but satisfied overall | Optional Wi-Fi upgrades for premium travelers |
| 3 | 45 | Long-haul Business | Very High | Poor departure/arrival convenience | Improve scheduling flexibility |
| 4 | 38 | Business | Moderate | Extreme delays impact satisfaction | Optimize route scheduling |
| 5 | 46 | Business & Personal | Low | Poor baggage handling, inflight service | Improve service reliability |

## 📌 Challenges & Next Steps
### 🔴 Challenges
- **Overlapping clusters** in PCA visualizations suggest multi-dimensional relationships.
- **Data imbalance** (economy vs. business travelers) affecting cluster formation.
- **Delay times** highly skewed, requiring better outlier handling.

### 🟢 Next Steps
- Apply **Hierarchical Clustering** for deeper segmentation.
- Enrich dataset with **route-specific** and **loyalty** program data.
- Implement **segment-specific recommendations** to improve airline customer satisfaction.

## 📜 Author
👤 **Wang Yi Ming**  
🎓 Master’s Student in Analytics & Visualisation  
🔗 [LinkedIn](https://www.linkedin.com/in/wangyiming)  

---

📌 **Want to contribute?** Feel free to fork, star ⭐, or suggest improvements! 🚀
