# Big Data and AI Trend Project: GridironIQ

**Note:** This project repository is created in partial fulfillment of the requirements for the Big Data Analytics course offered by the Master of Science in Business Analytics program at the Carlson School of Management, University of Minnesota.

**Executive Summary:** Our NFL Contract Valuation System aims to predict NFL contracts and guaranteed values for the offensive skilled position groups QB/RB/WR/TE by analyzing player performace statistics alongside historical contract data. We designed the dashboard keeping in mind NFL front offices, sports media analysts and fans that want to see how players are valued relative to the market. Our interactive dashboard includes functionality to predict contracts, compare players, analyze team contracts and enter smart queries such as "Show me undervalued RBs making $5M or less". 

Key takeaways from our research: 
1. Dallas Cowboys QB Dak Prescott’s contract sits $15M above our model’s predicted value. The team has not made the playoffs since, showing a direct consequence of the cap space lost to an overpriced QB. Our tool could help flag this before signing.

2. The Los Angeles Rams are the most cap-efficient team in the league. The Baltimore Ravens and Jacksonville Jaguars have the worst gaps, pointing to overspending across their roster.

Full information about our project can be found in our flier: https://github.com/zhan9921-afk/Big-Data-and-AI-Trend-Project/blob/main/GRIDIRONIQ_flier.pdf

The individual components for our interactive dashboard can be found here: https://github.com/zhan9921-afk/Big-Data-and-AI-Trend-Project/tree/main/Dashboard

**Pipeline overview, script and dataset:** Our pipeline follows a structured five-stage process designed to transform raw data into actionable insights. It begins with data ingestion, where multiple sources such as roster information, seasonal data and player contract data are collected into a centralized environment. The pipeline then moves into feature engineering, where key variables and metrics are derived to better represent player performance. We then run XGBoost and Classification techniques to obtain predicted vs actual contract information and generate player and roster grades. In the fourth stage, we apply a LLM layer to provide explanations for our predictions. Finally, the pipeline culminates in our interactive dashboards to enable decision-makers to easily interpret and act on the results.
  
  Our Pipeline requires the nflreadpy python package to be installed before running the model pipeline. The full pipeline can be found here: https://github.com/zhan9921-afk/Big-Data-and-AI-Trend-Project/blob/main/complete_modeling_pipeline.ipynb

  Our Pipeline is comprised of 12 models, 3 for each position group, to predict next contract length, guarenteed money and annual salary. The individual models can be found here: https://github.com/zhan9921-afk/Big-Data-and-AI-Trend-Project/tree/main/Dashboard/models

  The final dataset used for our pipeline can be found here: https://github.com/zhan9921-afk/Big-Data-and-AI-Trend-Project/blob/main/complete_final_data.csv
