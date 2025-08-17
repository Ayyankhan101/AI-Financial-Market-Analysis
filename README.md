# AI Financial Market Analysis Dashboard

This project provides a comprehensive analysis of the financial performance of major AI companies (OpenAI, Google, and Meta) using a synthetic dataset. The analysis is presented through an interactive web dashboard built with Streamlit.

## Dashboard Features

The interactive dashboard (`dashboard.py`) allows users to:

*   **Filter Data:** Select one or more companies and a specific date range to analyze.
*   **View Key Performance Indicators (KPIs):** See the latest AI revenue, R&D spending, average revenue growth, and the number of significant events.
*   **Visualize Time-Series Data:** An interactive line chart displays R&D spending and AI revenue over time.
*   **Analyze Events:** A scatter plot shows the relationship between AI revenue growth and stock impact during specific events.
*   **Monthly Heatmap:** A heatmap of monthly averages for key metrics provides a high-level overview of performance.
*   **Forecast Future Revenue:** A 90-day AI revenue forecast is generated using the Prophet forecasting model.
*   **Analyze Growth Trends:** A smoothed trend line of AI revenue growth helps to identify long-term patterns.
*   **Download Data:** The cleaned and filtered data can be downloaded as a CSV file.

## Dataset

The project uses a synthetic dataset named `ai_financial_market_daily_realistic_synthetic.csv`. This dataset was generated to simulate the daily financial and market data for the selected AI companies. The data includes the following columns:

*   `Date`: The date of the record.
*   `Company`: The name of the company (OpenAI, Google, Meta).
*   `R&D_Spending_USD_Mn`: Research and Development spending in millions of USD.
*   `AI_Revenue_USD_Mn`: Revenue from AI-related products and services in millions of USD.
*   `AI_Revenue_Growth_%`: The percentage growth of AI revenue.
*   `Event`: Descriptions of significant events that could impact the company's performance.
*   `Stock_Impact_%`: The percentage impact of an event on the company's stock price.

## Technologies Used

*   **Python:** The core programming language for data analysis and dashboard development.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Streamlit:** To create the interactive web dashboard.
*   **Plotly:** For creating interactive and visually appealing charts.
*   **Prophet:** For time-series forecasting.
*   **Jupyter Notebook:** For exploratory data analysis and model development.

## How to Run the Dashboard

1.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit dashboard:**
    ```bash
    streamlit run dashboard.py
    ```

    This will open the dashboard in your web browser.

## Analysis

The `Analysis.ipynb` Jupyter Notebook contains the detailed exploratory data analysis (EDA) and the steps taken to clean the data, generate insights, and build the forecasting model. The notebook provides a step-by-step guide to understanding the dataset and the analysis performed before building the dashboard.
