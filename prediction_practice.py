import pandas as pd
import numpy as np
import streamlit as st


#Machines setting
machines = pd.DataFrame({
    'MachineID': [1, 2, 3, 4, 5],
    'MachineName': ['A', 'B', 'C', 'D', 'E'],
    'AvgOutput': [100, 120, 80, 140, 90],
    'DefectRate': [0.05, 0.07, 0.08, 0.09, 0.04],
    'MachineAge': [5, 3, 8, 2, 7],
    'OperatorAbility': [0.9, 0.85, 0.7, 0.85, 0.95],
})

#Products
products = pd.DataFrame({
    'ProductID': [1, 2, 3],
    'ProductName': ['knobs', 'staples', 'tubes'],
    'BaseOutputModifier': [0.8, 1.0, 0.5],
})

#Cost
unit_price = {'knobs':2.0, 'staples':1.0, 'tubes':3.0}
defect_cost = {'knobs':0.5, 'staples':0.2, 'tubes':1.0}
machine_cost = {'A':10, 'B':12, 'C':8, 'D':14, 'E':9}

#streamlit user inputs
st.title('Interactive Predictive Quality Control Simulation')

#select products
selected_products = st.multiselect(
    'Select Products',
    products['ProductName'].tolist(),
    default=['knobs'],
)

#select number of days
days = st.slider("Number of days to simulate", min_value=10, max_value=90, step=1)



#Simulation
production_records = []

selected_products_df = products[products['ProductName'].isin(selected_products)]

for day in range(1, days + 1):
    for _, m in machines.iterrows():
        for _, p in selected_products_df.iterrows():
            #Produced units
            produced = max(0, int(np.random.normal(float(m['AvgOutput']) * float(p['BaseOutputModifier']), 5)))

            #Defect factor
            defect_factor = (1 + 0.05*float(m['MachineAge'])) * (1 - float(m['OperatorAbility']))
            #defects = np.random.binomial(produced, min(float(m['DefectRate']) * defect_factor, 1))

            # Introduce a systematic trend: defect rate slightly increases over days
            trend_factor = 1 + 0.05 * day  # 5% increase per day
            env_factor = np.random.uniform(0.9, 1.1)  # random daily effect

            defects = np.random.binomial(
                produced, min(float(m['DefectRate']) * trend_factor * env_factor, 1)
            )

            #Economics part
            revenue = produced * unit_price[p['ProductName']]
            cost_of_defects = defects * defect_cost[p['ProductName']]
            operating_cost = machine_cost[m['MachineName']]
            profit = revenue - (cost_of_defects + operating_cost)

            production_records.append([day, m['MachineName'], p['ProductName'], produced, defects,revenue, cost_of_defects, operating_cost, profit])


#create dataframe
production_df = pd.DataFrame(
    production_records,
    columns=['Day', 'Machine', 'Product', 'Production', 'Defects', 'Revenue', 'Cost of Defects', 'Operating Cost', 'Profit'],
)
production_df['DefectRate'] = production_df['Defects']/production_df['Production']

#calculate the predicted defect rate
production_df['PredictedDefectRate'] = production_df.groupby(['Machine', 'Product'])['DefectRate'].transform(lambda x: x.rolling(3, min_periods=1).mean())


#display visualization
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader('Defect Rate Over Time')

for product in selected_products:
    st.write(f"**{product}**")
    df_plot = production_df[production_df['Product'] == product].pivot(index='Day', columns='Machine', values='DefectRate')
    st.line_chart(df_plot)

st.subheader("Predicted Defect Rate (5-day Moving Average)")

for product in selected_products:
    st.write(f"**{product}**")
    df_pred = production_df[production_df['Product'] == product].pivot(index='Day', columns='Machine', values='PredictedDefectRate')
    st.line_chart(df_pred)

st.subheader("Total Defects by Machine & Product")
total_defects = production_df.groupby(['Machine', 'Product'])['Defects'].sum().unstack().fillna(0)
st.bar_chart(total_defects)


from sklearn.linear_model import LinearRegression
st.subheader("Defect Rate Trend Analysis (Regression)")

for product in selected_products:
    st.write(f"**{product}**")
    df_prod = production_df[production_df['Product'] == product]

    X = df_prod['Day'].values.reshape(-1, 1)
    y = df_prod['DefectRate'].values

    model = LinearRegression()
    model.fit(X, y)

    # Predicted trend line
    y_pred = model.predict(X)

    # Plot actual vs regression
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df_prod['Day'], df_prod['DefectRate'], color='blue', label='Actual')
    ax.plot(df_prod['Day'], y_pred, color='red', label='Trend')
    ax.set_xlabel('Day')
    ax.set_ylabel('Defect Rate')
    ax.set_title(f"Defect Rate Trend for {product}")
    ax.legend()

    st.pyplot(fig)

    # Optional: show slope
    st.write(f"Estimated daily change in defect rate: {model.coef_[0]:.4f}")


st.subheader("Machine-Level Summary Report")

# Aggregate production and defects by Machine and Product
machine_summary = production_df.groupby(['Machine', 'Product']).agg(
    Total_Produced=('Production', 'sum'),
    Total_Defects=('Defects', 'sum'),
)

# Calculate defect rate
machine_summary['Overall_DefectRate'] = machine_summary['Total_Defects'] / machine_summary['Total_Produced']

# Reset index for better display
machine_summary = machine_summary.reset_index()

st.dataframe(machine_summary)


# Economics Charts
# =========================
st.subheader("Total Profit by Machine & Product")
profit_summary = production_df.groupby(['Machine','Product'])['Profit'].sum().unstack().fillna(0)
st.bar_chart(profit_summary)

st.subheader("Revenue vs Defect Cost by Product")
rev_defects = production_df.groupby('Product')[['Revenue','Cost of Defects']].sum()
st.bar_chart(rev_defects)


#show data
st.subheader("Production Data")
st.dataframe(production_df)


#streamlit run <path_to_file>\prediction_practice.py




