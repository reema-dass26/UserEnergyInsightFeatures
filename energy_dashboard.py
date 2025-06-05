import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import altair as alt

# Page setup
st.set_page_config(page_title="Energy Insights Dashboard", layout="wide")

@st.cache_data
def load_data(filepath):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    df.rename(columns={'timestamp': 'time', 'main': 'consumption_kwh'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    df['consumption_kwh'] = pd.to_numeric(df['consumption_kwh'], errors='coerce')
    return df.dropna()

# Load data
data_path = "data/synthetic_1_year.xlsx"
if not os.path.exists(data_path):
    st.error("Data file not found. Please make sure the file is in the 'data/' directory.")
    st.stop()

df = load_data(data_path)
df['date'] = df['time'].dt.date
daily_summary = df.groupby('date')['consumption_kwh'].sum().reset_index()

# Tabs
tab_titles = [
    "📅 Daily Usage Summary",
    "📊 Weekly Trends",
    "⏰ High Usage Hours",
    "📆 Weekday vs Weekend",
    "🌤 Seasonal Energy Tracker",
    "🌙 Phantom Load Detector",
    "💰 Energy Budget Planner",
    "🏅 Consumption Milestones",
    "🌍 Carbon Footprint Estimator",
    "🏁 Power Consumption Challenge",
    "🚨 Smart Alerts",
    " 🌦️ Weather Energy Overlay"
]

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(tab_titles)

with tab1:
    st.markdown("## 🔋 Daily Energy Usage Summary")
    st.markdown("Easily track your daily electricity usage and stay on top of your consumption.")

    # Date filter
    min_date = daily_summary['date'].min()
    max_date = daily_summary['date'].max()

    start_date, end_date = st.slider(
        "Select a date range to view:",
        min_value=min_date,
        max_value=max_date,
        value=(max_date - pd.Timedelta(days=30), max_date),
        format="DD/MM/YYYY"
    )

    filtered_df = daily_summary[
        (daily_summary['date'] >= start_date) & (daily_summary['date'] <= end_date)
    ]

    # Key Metrics
    total_kwh = filtered_df['consumption_kwh'].sum()
    avg_kwh = filtered_df['consumption_kwh'].mean()

    col1, col2 = st.columns(2)
    col1.metric("🔌 Total Energy Used", f"{total_kwh:.2f} kWh")
    col2.metric("📈 Average Daily Use", f"{avg_kwh:.2f} kWh")

    # Color logic for bars
    def get_bar_color(val, avg):
        if val > avg * 1.2:
            return '#d62728'  # red for high
        elif val < avg * 0.8:
            return '#2ca02c'  # green for low
        else:
            return '#1f77b4'  # blue for average

    bar_colors = [get_bar_color(v, avg_kwh) for v in filtered_df['consumption_kwh']]

    # # Plot
    # fig, ax = plt.subplots(figsize=(12, 5))
    # ax.bar(filtered_df['date'], filtered_df['consumption_kwh'], color=bar_colors)
    # ax.axhline(avg_kwh, color='orange', linestyle='--', label='Average Usage')
    # ax.set_title("Daily Energy Usage", fontsize=14)
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Energy (kWh)")
    # ax.grid(axis='y', linestyle='--', alpha=0.5)
    # ax.legend()
    # fig.autofmt_xdate(rotation=45)
    # st.pyplot(fig)
    # Prepare daily summary
    df['date'] = df['time'].dt.date
    daily_summary = df.groupby('date')['consumption_kwh'].sum().reset_index()

    # Filtered range
    filtered_df = daily_summary[
        (daily_summary['date'] >= start_date) & (daily_summary['date'] <= end_date)
    ]

    # Altair Chart
    daily_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('consumption_kwh:Q', title='Energy (kWh)'),
        tooltip=['date:T', 'consumption_kwh:Q']
    ).properties(
        title='📅 Daily Energy Usage',
        width='container',
        height=300
    )

    st.altair_chart(daily_chart, use_container_width=True)
    with st.expander("📋 View Daily Data Table"):
            st.dataframe(filtered_df, use_container_width=True)

with tab2:
    st.markdown("## 📅 Weekly Energy Trends")
    st.markdown("Compare how your energy consumption varies week to week.")

    # Prepare weekly data
    df['week'] = df['time'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_summary = df.groupby('week')['consumption_kwh'].sum().reset_index()

    # Determine average, thresholds and colors
    mean_val = weekly_summary['consumption_kwh'].mean()
    high_thresh = mean_val * 1.05
    low_thresh = mean_val * 0.95

    def classify(value):
        if value >= high_thresh:
            return '#d62728'  # red for high
        elif value <= low_thresh:
            return '#2ca02c'  # green for low
        else:
            return '#1f77b4'  # blue for average

    weekly_summary['color'] = weekly_summary['consumption_kwh'].apply(classify)

    # Show latest N weeks
    show_weeks = st.slider("Select number of weeks to display:", 4, len(weekly_summary), 12)
    weekly_display = weekly_summary.tail(show_weeks).copy()

    # Add label column for Altair (e.g., '24-04')
    weekly_display['label'] = weekly_display['week'].dt.strftime('%y-%m')

    # Altair plot
    st.markdown("### Weekly Usage Chart")
    chart = alt.Chart(weekly_display).mark_bar().encode(
        x=alt.X('label:N', title="Week (YY-MM)"),
        y=alt.Y('consumption_kwh:Q', title='Energy (kWh)'),
        color=alt.Color('color:N', scale=None, legend=None),
        tooltip=[
            alt.Tooltip('week:T', title='Week Start'),
            alt.Tooltip('consumption_kwh:Q', title='Total kWh', format='.2f')
        ]
    ).properties(
        width=700,
        height=400,
        title="📊 Weekly Energy Usage with Hover Insights"
    ).configure_axisX(labelAngle=45)

    st.altair_chart(chart, use_container_width=True)

    # Weekly data table
    with st.expander("📋 Weekly Data Table"):
        st.dataframe(weekly_display[['week', 'consumption_kwh']], use_container_width=True)
with tab3:
    st.markdown("## ⏰ High Usage Hours")
    st.markdown("Find out which hours of the day consume the most energy on average.")

    # Extract hour and average across all days
    df['hour'] = df['time'].dt.hour
    hourly_avg = df.groupby('hour')['consumption_kwh'].mean().reset_index()

    # Highlight top usage hours
    top_hours = hourly_avg.sort_values(by='consumption_kwh', ascending=False).head(3)
    avg_val = hourly_avg['consumption_kwh'].mean()

    # Chart
        # Prepare color-coded data
    hourly_avg['highlight'] = hourly_avg['hour'].apply(
        lambda h: '🔥 High' if h in top_hours['hour'].values
        else ('🟢 Low' if hourly_avg.loc[hourly_avg['hour'] == h, 'consumption_kwh'].values[0] < avg_val * 0.95
                else '🔵 Average')
        )

        # Altair chart
    hourly_chart = alt.Chart(hourly_avg).mark_bar().encode(
            x=alt.X('hour:O', title='Hour of Day'),
            y=alt.Y('consumption_kwh:Q', title='Avg Energy (kWh)'),
            color=alt.Color('highlight:N', scale=alt.Scale(
                domain=['🔥 High', '🟢 Low', '🔵 Average'],
                range=['#d62728', '#2ca02c', '#4e79a7']
            )),
            tooltip=['hour', 'consumption_kwh']
        ).properties(
            title="⏱️ Hourly Energy Patterns",
            width='container',
            height=300
        )

    st.altair_chart(hourly_chart, use_container_width=True)

    with st.expander("📋 View Hourly Averages"):
        st.dataframe(hourly_avg.rename(columns={'hour': 'Hour of Day', 'consumption_kwh': 'Avg kWh'}), use_container_width=True)
with tab4:
    st.markdown("## Weekday vs Weekend")
    st.markdown("Understand your energy habits hour-by-hour. See what times you use the most — and compare weekdays vs weekends.")

    df['hour'] = df['time'].dt.hour
    df['day_type'] = df['time'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

    hourly_group = df.groupby(['day_type', 'hour'])['consumption_kwh'].mean().reset_index()

    # User-defined goal
    st.markdown("### 🎯 Optional: Set a Max Target for Hourly Usage")
    goal = st.slider("Set your hourly kWh goal:", 0.1, 1.0, 0.6, step=0.05)

    # Plot
    st.markdown("### Average Hourly Consumption – Weekday vs Weekend")
    import altair as alt

    # Prepare data
    goal_df = pd.DataFrame({
        'hour': list(range(24)),
        'goal': [goal] * 24
    })

    # Base chart with grouped line per day type
    line_chart = alt.Chart(hourly_group).mark_line(point=True).encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('consumption_kwh:Q', title='Avg Energy (kWh)'),
        color=alt.Color('day_type:N', scale=alt.Scale(
            domain=['Weekday', 'Weekend'],
            range=['#1f77b4', '#ff7f0e']
        )),
        tooltip=['day_type:N', 'hour:O', 'consumption_kwh:Q']
    )

    # Goal line overlay
    goal_line = alt.Chart(goal_df).mark_rule(strokeDash=[4,2], color='red').encode(
        x='hour:O',
        y='goal:Q'
    )

    # Combine and render
    combined_chart = (line_chart + goal_line).properties(
        title="Hourly Energy Use: Weekdays vs Weekends",
        width='container',
        height=300
    )

    st.altair_chart(combined_chart, use_container_width=True)


    # Expandable raw data
    with st.expander("📋 View Hourly Averages by Day Type"):
        st.dataframe(hourly_group.pivot(index='hour', columns='day_type', values='consumption_kwh').round(3), use_container_width=True)


with tab5:
    st.markdown("## 🌤 Seasonal Energy Tracker")
    st.markdown("Visualize and compare your energy consumption across different seasons.")

    # Assign seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['time'].dt.month.apply(get_season)
    seasonal_summary = df.groupby('season')['consumption_kwh'].sum().reindex(['Winter', 'Spring', 'Summer', 'Fall']).reset_index()


    # Color map
    season_colors = {
        'Winter': '#1f77b4',
        'Spring': '#2ca02c',
        'Summer': '#ff7f0e',
        'Fall': '#8c564b'
    }

    seasonal_summary['color'] = seasonal_summary['season'].map(season_colors)

    # Altair chart
    chart = alt.Chart(seasonal_summary).mark_bar().encode(
        x=alt.X('season:N', title='Season'),
        y=alt.Y('consumption_kwh:Q', title='Total Energy (kWh)'),
        color=alt.Color('season:N', scale=alt.Scale(domain=list(season_colors.keys()), range=list(season_colors.values()))),
        tooltip=['season:N', 'consumption_kwh:Q']
    ).properties(
        title='Total Energy Usage by Season',
        width='container',
        height=300
    )

    st.altair_chart(chart, use_container_width=True)


    # Show exact values
    with st.expander("📋 View Seasonal Totals"):
        st.dataframe(seasonal_summary.rename(columns={'consumption_kwh': 'Total kWh'}), use_container_width=True)
with tab6:
    st.markdown("## 🌙 Phantom Load Detector")
    st.markdown("Spot and track unusually high energy use during idle hours (1 AM – 5 AM).")

    # Filter for idle hours
    df['hour'] = df['time'].dt.hour
    df['date'] = df['time'].dt.date
    phantom_df = df[(df['hour'] >= 1) & (df['hour'] <= 5)]

    phantom_summary = phantom_df.groupby('date')['consumption_kwh'].sum().reset_index()
    avg_phantom = phantom_summary['consumption_kwh'].mean()

    # Flag days with high idle usage
    phantom_summary['status'] = phantom_summary['consumption_kwh'].apply(
        lambda x: '⚠️ High' if x > avg_phantom * 1.2 else '✅ Normal'
    )

    # Altair Chart
    phantom_chart = alt.Chart(phantom_summary).mark_bar().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('consumption_kwh:Q', title='Energy (kWh)'),
        color=alt.condition(
            alt.datum.status == '⚠️ High',
            alt.value('orange'),  # High phantom load
            alt.value('#1f77b4')  # Normal
        ),
        tooltip=['date:T', 'consumption_kwh:Q', 'status:N']
    ).properties(
        title='Phantom Load (1 AM – 5 AM)',
        width='container',
        height=300
    )

    st.altair_chart(phantom_chart, use_container_width=True)

    with st.expander("📋 View Phantom Load Table"):
        st.dataframe(phantom_summary, use_container_width=True)


with tab7:
    st.markdown("## 💰 Energy Budget Planner")
    st.markdown("Plan your energy use and set a realistic budget based on past trends.")

    # Monthly aggregation
    df['month'] = df['time'].dt.to_period('M').dt.to_timestamp()
    monthly_summary = df.groupby('month')['consumption_kwh'].sum().reset_index()

    # Show recent months
    months_available = len(monthly_summary)
    lookback = st.slider("Lookback window for average (months):", 1, min(6, months_available), 3)

    recent_avg = monthly_summary.tail(lookback)['consumption_kwh'].mean()
    st.markdown(f"📊 **Avg Monthly Usage (Last {lookback} Months):** `{recent_avg:.2f} kWh`")

    # User-defined or suggested budget
    suggested = round(recent_avg * 0.95, 2)
    user_budget = st.number_input("Set your monthly energy budget (kWh):", min_value=0.0, value=suggested)

    # Label each month as over/under budget
    monthly_summary['budget_status'] = monthly_summary['consumption_kwh'].apply(
        lambda x: 'Over Budget' if x > user_budget else 'Within Budget'
    )

    # Altair bar chart
    budget_chart = alt.Chart(monthly_summary).mark_bar().encode(
        x=alt.X('month:T', title='Month'),
        y=alt.Y('consumption_kwh:Q', title='Energy (kWh)'),
        color=alt.condition(
            alt.datum.budget_status == 'Over Budget',
            alt.value('#e74c3c'),  # Red
            alt.value('#2ecc71')   # Green
        ),
        tooltip=['month:T', 'consumption_kwh:Q', 'budget_status:N']
    ).properties(
        title='Monthly Usage vs Budget',
        width='container',
        height=300
    )

    st.altair_chart(budget_chart, use_container_width=True)

    with st.expander("📋 Monthly Budget Table"):
        st.dataframe(monthly_summary[['month', 'consumption_kwh', 'budget_status']], use_container_width=True)


with tab8:
    st.markdown("## 🏅 Consumption Milestones")
    st.markdown("Celebrate your most efficient days and spot high-usage patterns.")

    # Daily totals
    daily_kwh = df.groupby(df['time'].dt.date)['consumption_kwh'].sum().reset_index()
    daily_kwh.columns = ['date', 'consumption_kwh']

    # Personal bests
    lowest_day = daily_kwh.loc[daily_kwh['consumption_kwh'].idxmin()]
    highest_day = daily_kwh.loc[daily_kwh['consumption_kwh'].idxmax()]
    average_day = daily_kwh['consumption_kwh'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Lowest Day", f"{lowest_day['consumption_kwh']:.2f} kWh", str(lowest_day['date']))
    col2.metric("📈 Highest Day", f"{highest_day['consumption_kwh']:.2f} kWh", str(highest_day['date']))
    col3.metric("📊 Average Daily Use", f"{average_day:.2f} kWh")

    # --- Bar Chart for Milestones ---
    milestone_data = pd.DataFrame({
        'Day': ['Lowest', 'Highest', 'Average'],
        'kWh': [
            lowest_day['consumption_kwh'],
            highest_day['consumption_kwh'],
            average_day
        ]
    })

    bar_chart = alt.Chart(milestone_data).mark_bar(size=60).encode(
        x=alt.X('Day:N', title='Milestone Type'),
        y=alt.Y('kWh:Q', title='Consumption (kWh)'),
        color=alt.Color('Day:N', scale=alt.Scale(domain=['Lowest', 'Highest', 'Average'],
                                                 range=['#2ca02c', '#d62728', '#1f77b4'])),
        tooltip=['Day', 'kWh']
    ).properties(
        width=500,
        height=400,
        title="📊 Milestone Day Comparison"
    )

    st.altair_chart(bar_chart, use_container_width=True)

    # --- Additional Insight 1: Streak Analysis ---
    from itertools import groupby

    daily_kwh['above_avg'] = daily_kwh['consumption_kwh'] > average_day
    daily_kwh['below_avg'] = daily_kwh['consumption_kwh'] < average_day

    def get_streaks(series, condition=True):
        return max((sum(1 for _ in group) for key, group in groupby(series) if key == condition), default=0)

    low_streak = get_streaks(daily_kwh['below_avg'])
    high_streak = get_streaks(daily_kwh['above_avg'])

    col4, col5 = st.columns(2)
    col4.metric("💚 Longest Efficient Streak", f"{low_streak} days")
    col5.metric("💢 Longest High Usage Streak", f"{high_streak} days")

    # --- Additional Insight 2: Weekday vs Weekend Comparison ---
    daily_kwh['weekday'] = pd.to_datetime(daily_kwh['date']).dt.dayofweek
    daily_kwh['is_weekend'] = daily_kwh['weekday'] >= 5

    weekend_avg = daily_kwh[daily_kwh['is_weekend']]['consumption_kwh'].mean()
    weekday_avg = daily_kwh[~daily_kwh['is_weekend']]['consumption_kwh'].mean()

    st.markdown("### 🗓️ Weekday vs Weekend Usage")
    st.write(f"**Weekday Avg:** {weekday_avg:.2f} kWh  &nbsp;|&nbsp;  **Weekend Avg:** {weekend_avg:.2f} kWh")

    # --- Additional Insight 3: Most Volatile Day ---
    daily_kwh['pct_change'] = daily_kwh['consumption_kwh'].pct_change() * 100
    max_spike = daily_kwh.loc[daily_kwh['pct_change'].abs().idxmax()]
    st.metric("⚡ Most Volatile Day", f"{max_spike['consumption_kwh']:.2f} kWh", f"{max_spike['pct_change']:+.1f}%")

    # --- All Daily Records ---
    with st.expander("📋 View All Daily Records"):
        st.dataframe(daily_kwh.sort_values(by='consumption_kwh'), use_container_width=True)

    # --- Histogram with Color-Coded Ranges ---
    st.markdown("### 📈 Daily Consumption Distribution")

    average_day = daily_kwh['consumption_kwh'].mean()

    # Define consumption ranges
    def label_range(kwh):
        if kwh < average_day * 0.9:
            return "Below Avg"
        elif kwh > average_day * 1.1:
            return "Above Avg"
        else:
            return "Around Avg"

    daily_kwh['range_label'] = daily_kwh['consumption_kwh'].apply(label_range)

    # Build histogram with colored bins
    hist = alt.Chart(daily_kwh).mark_bar().encode(
        alt.X("consumption_kwh", bin=alt.Bin(maxbins=30), title="Daily Consumption (kWh)"),
        y=alt.Y("count()", title="Number of Days"),
        color=alt.Color("range_label:N", 
                        title="Range",
                        scale=alt.Scale(domain=["Below Avg", "Around Avg", "Above Avg"],
                                        range=["#1f77b4", "#2ca02c", "#d62728"])),
        tooltip=[
            alt.Tooltip("count()", title="Days"),
            alt.Tooltip("range_label:N", title="Range")
        ]
    ).properties(
        width=500,
        height=300,
        title="🔍 Distribution of Daily Usage (Colored by Range)"
    )

    # Optional: add vertical average line
    average_rule = alt.Chart(pd.DataFrame({'avg': [average_day]})).mark_rule(color='black', strokeDash=[4, 4]).encode(
        x='avg:Q'
    )

    st.altair_chart(hist + average_rule, use_container_width=True)

with tab9:
    st.markdown("## 🌍 Carbon Footprint Estimator")
    st.markdown("See how your electricity use translates into environmental impact.")

    # User-defined emission factor
    st.info("Default emission factor: **0.475 kg CO₂ / kWh** (global avg).")
    emission_factor = st.number_input(
        "Set emission factor (kg CO₂ / kWh)", min_value=0.0, value=0.475
    )

    # Calculate emissions
    df['date'] = df['time'].dt.date
    daily_emission = df.groupby('date')['consumption_kwh'].sum().reset_index()
    daily_emission['kg_co2'] = daily_emission['consumption_kwh'] * emission_factor
    total_emission = daily_emission['kg_co2'].sum()
    st.success(f"🧮 Estimated total CO₂ emissions: **{total_emission:.2f} kg**")

    # Educational: 5 Impacts
    with st.expander("🧠 Learn: 5 Major Impacts of CO₂ Emissions"):
        st.markdown("""
        1. 🌡️ **Global Warming** – CO₂ traps heat, causing rising temperatures.  
        2. 🌊 **Rising Sea Levels** – Melts glaciers, floods coasts.  
        3. 🏞️ **Deforestation Pressure** – Trees absorb CO₂. More emissions = fewer forests.  
        4. 🐝 **Biodiversity Loss** – Warmer climate disrupts habitats.  
        5. 💨 **Air Pollution Side Effects** – Indirect contributors to smog and health issues.
        """)

    # Area chart of CO₂ emissions
    carbon_chart = alt.Chart(daily_emission).mark_area(
        line={'color': '#1f77b4'}, color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='#a6cee3', offset=0), alt.GradientStop(color='#1f77b4', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('kg_co2:Q', title='Estimated CO₂ (kg)'),
        tooltip=['date:T', 'kg_co2:Q']
    ).properties(
        title='📈 Daily Carbon Emissions Trend',
        width='container',
        height=300
    )
    st.altair_chart(carbon_chart, use_container_width=True)

    # Best/Worst days
    best_day = daily_emission.loc[daily_emission['kg_co2'].idxmin()]
    worst_day = daily_emission.loc[daily_emission['kg_co2'].idxmax()]
    col1, col2 = st.columns(2)
    col1.metric("💚 Lowest CO₂ Day", f"{best_day['kg_co2']:.2f} kg", str(best_day['date']))
    col2.metric("💢 Highest CO₂ Day", f"{worst_day['kg_co2']:.2f} kg", str(worst_day['date']))

    # Monthly tree prompt + visuals
    from datetime import datetime
    last_month = datetime.now().month - 1 or 12
    monthly_emission = daily_emission[
        pd.to_datetime(daily_emission['date']).dt.month == last_month
    ]['kg_co2'].sum()
    trees_killed = int(monthly_emission / 21)

    from PIL import Image
    def load_resized_gif(path, max_width=700):
        try:
            gif = Image.open(path)
            ratio = max_width / gif.width
            new_size = (int(gif.width * ratio), int(gif.height * ratio))
            gif = gif.resize(new_size, Image.Resampling.LANCZOS)
            return gif
        except Exception as e:
            st.warning(f"Error loading or resizing GIF: {e}")
            return None

    gif_path = "images/carbon_impact_drama.gif"
    resized_gif = load_resized_gif(gif_path)

    show_gif = st.checkbox("🎞️ Show Dramatic CO₂ Impact GIF", value=True)

    if show_gif:
        col_left, col_right = st.columns([2, 1.5])
        st.markdown("## 🪵 Uh-oh... Tree Trouble?")
        st.markdown(f"Last month, your electricity usage may have cost us **🌳 {trees_killed} trees** worth of CO₂ absorption!")
        st.markdown("#### 😬 Time to give back to the planet?")

        with col_left:
            
            if gif_path:
                st.image(gif_path, caption="Every kWh adds up... 🌍💔")

        with col_right:
            st.markdown("### 📘 Quick Tips")
            # st.image("/mnt/data/80f8214c-3957-4c19-b200-b71fe86df0cf.png", caption="Small habits, big impact", use_container_width=True)
            st.markdown("""
            **🌱 Save Energy By:**
            - Unplug idle chargers  
            - Use LED lights  
            - Dry clothes naturally  
            - Turn off standby devices  
            - Set fridge temp smartly
            """)

    # Equivalent real-world breakdown
    with st.expander("🧾 What Else Does This Amount of CO₂ Equal?"):
        st.markdown(f"""
        - 🚗 **{monthly_emission / 0.404:.0f} miles** driven by a gas car  
        - ✈️ **{monthly_emission / 90:.1f} short-haul flights** (avg 90kg CO₂ per flight)  
        - 💡 **{monthly_emission / 0.92:.0f} hours** of LED lighting (avg 0.92 kg CO₂ / 1000 hrs)  
        - 🥩 **{monthly_emission / 27:.0f} steaks** worth of emissions  
        - 🧺 **{monthly_emission / 2.1:.0f} laundry loads** in a dryer
        """)

    # Data table
    with st.expander("📋 View Carbon Data"):
        st.dataframe(daily_emission, use_container_width=True)


with tab10:
    st.markdown("## 🏁 Power Consumption Challenge")
    st.markdown("See how you did this week compared to last week. Let's reduce energy step by step!")

    # Prepare weekly data
    df['week'] = df['time'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_kwh = df.groupby('week')['consumption_kwh'].sum().reset_index()

    if len(weekly_kwh) < 2:
        st.warning("Not enough data for comparison. Need at least two weeks.")
    else:
        last_two = weekly_kwh.tail(2)
        this_week = last_two.iloc[1]
        last_week = last_two.iloc[0]

        change = this_week['consumption_kwh'] - last_week['consumption_kwh']
        percent_change = (change / last_week['consumption_kwh']) * 100

        # Feedback logic
        if change < 0:
            feedback = f"🎉 Great job! You reduced your energy use by **{abs(percent_change):.2f}%**."
        elif change > 0:
            feedback = f"⚡️ You used **{percent_change:.2f}%** more energy this week. Let's aim lower next week!"
        else:
            feedback = "🔄 Same usage as last week. Try to improve next week!"

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("📅 Last Week", f"{last_week['consumption_kwh']:.2f} kWh")
        col2.metric("📅 This Week", f"{this_week['consumption_kwh']:.2f} kWh", f"{percent_change:+.2f}%")

        # Bar Chart Comparison (Altair)
        comparison_df = pd.DataFrame({
            'Week': ['Last Week', 'This Week'],
            'kWh': [last_week['consumption_kwh'], this_week['consumption_kwh']],
            'Color': ['#1f77b4', '#2ca02c' if change <= 0 else '#d62728']
        })

        comparison_chart = alt.Chart(comparison_df).mark_bar().encode(
            x=alt.X('Week:N', title=None),
            y=alt.Y('kWh:Q', title='Energy (kWh)'),
            color=alt.Color('Color:N', scale=None, legend=None),
            tooltip=['Week', 'kWh']
        ).properties(
            title='📊 Weekly Usage Comparison',
            width='container',
            height=300
        )

        st.altair_chart(comparison_chart, use_container_width=True)
        st.success(feedback)

        # Divider and goal tracker
        st.markdown("---")
        st.markdown("### 🎯 Set Your Weekly Energy Goal")

        default_goal = round(last_week['consumption_kwh'], 2)
        goal_kwh = st.number_input(
            "Enter your energy goal for this week (kWh):",
            min_value=0.0,
            value=default_goal,
            step=0.1
        )

        # Goal comparison logic
        goal_diff = this_week['consumption_kwh'] - goal_kwh
        goal_percent = (goal_diff / goal_kwh) * 100

        if goal_diff < 0:
            st.success(f"🏅 You're **{abs(goal_percent):.2f}% below** your goal. Keep it up!")
        elif goal_diff > 0:
            st.warning(f"⚠️ You're **{goal_percent:.2f}% over** your goal. Adjust your usage next week!")
        else:
            st.info("⏸️ You've exactly met your goal this week!")

        # Goal Comparison Chart
        goal_df = pd.DataFrame({
            'Category': ['Goal', 'This Week'],
            'kWh': [goal_kwh, this_week['consumption_kwh']],
            'Color': ['#8c564b', '#2ca02c' if goal_diff <= 0 else '#d62728']
        })

        goal_chart = alt.Chart(goal_df).mark_bar().encode(
            x=alt.X('Category:N', title=None),
            y=alt.Y('kWh:Q', title='Energy (kWh)'),
            color=alt.Color('Color:N', scale=None, legend=None),
            tooltip=['Category', 'kWh']
        ).properties(
            title='🎯 This Week vs. Your Goal',
            width='container',
            height=300
        )

        st.altair_chart(goal_chart, use_container_width=True)
with tab11:
    st.markdown("## 🚨 Smart Alerts: Unusual Spike Detection")
    st.markdown("Get notified about sudden energy spikes that may signal unusual appliance use or billing issues.")

    # Resample hourly to ensure rolling window resolution is consistent
    hourly_df = df.set_index('time').resample('1H')['consumption_kwh'].sum().reset_index()

    # 24-hour rolling average and standard deviation
    hourly_df['rolling_mean'] = hourly_df['consumption_kwh'].rolling(window=24).mean()
    hourly_df['rolling_std'] = hourly_df['consumption_kwh'].rolling(window=24).std()

    # Define spike as value greater than mean + 2 * std
    hourly_df['is_spike'] = (
        hourly_df['consumption_kwh'] >
        (hourly_df['rolling_mean'] + 4 * hourly_df['rolling_std'])
    )

    # Extract spike events
    spikes = hourly_df[hourly_df['is_spike']].dropna()

    if not spikes.empty:
        st.warning(f"⚠️ {len(spikes)} unusual spike(s) detected in your consumption data.")

        # Display spikes as a chart
        base = alt.Chart(hourly_df).mark_line().encode(
            x='time:T',
            y='consumption_kwh:Q',
            tooltip=['time:T', 'consumption_kwh']
        ).properties(title="Hourly Consumption with Spikes", width='container', height=300)

        spike_overlay = alt.Chart(spikes).mark_circle(size=60, color='red').encode(
            x='time:T',
            y='consumption_kwh:Q',
            tooltip=['time:T', 'consumption_kwh']
        )

        st.altair_chart(base + spike_overlay, use_container_width=True)

        with st.expander("📋 View Spike Details"):
            st.dataframe(spikes[['time', 'consumption_kwh']], use_container_width=True)
    else:
        st.success("✅ No significant spikes detected in the selected data.")


    # Simulated in-app alert
    if not spikes.empty:
        last_spike_time = spikes['time'].max()
        st.toast(f"⚡ Alert: High energy spike detected at {last_spike_time.strftime('%Y-%m-%d %H:%M')}!", icon="🚨")

    # Heatmap visualization
    st.markdown("### 🔥 Heatmap: Hourly Energy Usage")
    hourly_df['hour'] = hourly_df['time'].dt.hour
    hourly_df['date'] = hourly_df['time'].dt.date

    heatmap_data = hourly_df.groupby(['date', 'hour'])['consumption_kwh'].sum().reset_index()

    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('date:T', title='Date'),
        color=alt.Color('consumption_kwh:Q', scale=alt.Scale(scheme='reds'), title='kWh'),
        tooltip=['date:T', 'hour:O', 'consumption_kwh:Q']
    ).properties(
        width='container',
        height=400,
        title='Hourly Consumption Heatmap'
    )

    st.altair_chart(heatmap, use_container_width=True)

import numpy as np

# Add simulated weather data if missing
if 'temperature_c' not in df.columns:
    df['date'] = df['time'].dt.date
    unique_dates = df['date'].unique()

    # Generate fake temperatures (seasonal trend + noise)
    days = np.arange(len(unique_dates))
    base_temp = 20 + 10 * np.sin(2 * np.pi * days / 365)
    noise = np.random.normal(0, 3, len(unique_dates))
    simulated_temps = base_temp + noise

    # Create DataFrame
    temp_df = pd.DataFrame({
        'date': unique_dates,
        'temperature_c': simulated_temps
    })

    # Merge with main df
    df = df.merge(temp_df, on='date', how='left')
with tab12:
    st.markdown("## ⛅ Weather–Energy Overlay")
    st.markdown("Compare your energy spikes with temperature data to understand outside influences.")

    # Prepare daily summary
    daily_df = df.groupby('date').agg({
        'consumption_kwh': 'sum',
        'temperature_c': 'mean'
    }).reset_index()

    # Base chart
    base = alt.Chart(daily_df).encode(x='date:T')

    # 🔵 Energy usage line
    line_energy = base.mark_line(color='#1f77b4').encode(
        y=alt.Y('consumption_kwh:Q', axis=alt.Axis(title='Energy (kWh)')),
        tooltip=['date:T', 'consumption_kwh']
    )

    # 🔴 Temperature line
    line_temp = base.mark_line(color='#e377c2').encode(
        y=alt.Y('temperature_c:Q', axis=alt.Axis(title='Temperature (°C)')),
        tooltip=['date:T', 'temperature_c']
    )

    # Overlay both lines
    chart = alt.layer(line_energy, line_temp).resolve_scale(
        y='independent'
    ).properties(
        title="📊 Temperature vs Energy Consumption",
        width='container',
        height=350
    )

    st.altair_chart(chart, use_container_width=True)

    # Optional table
    with st.expander("📋 View Daily Comparison Table"):
        st.dataframe(daily_df, use_container_width=True)
    # --- Insight 1: Correlation between temperature and energy usage
    correlation = daily_df['temperature_c'].corr(daily_df['consumption_kwh'])
    st.markdown(f"### 📊 Temperature–Energy Correlation")
    st.write(f"Your energy usage and temperature have a correlation of **{correlation:.2f}**.")
    if correlation > 0.3:
        st.info("You tend to use more energy on hotter days — likely due to cooling (AC).")
    elif correlation < -0.3:
        st.info("You tend to use more energy on colder days — likely due to heating.")
    else:
        st.info("No strong correlation — your usage is likely not weather-driven.")

    # --- Insight 2: Comfort range detection
    temp_bins = pd.cut(daily_df['temperature_c'], bins=[-10, 15, 22, 30, 45], labels=['Cold', 'Comfort', 'Warm', 'Hot'])
    avg_by_temp_zone = daily_df.groupby(temp_bins)['consumption_kwh'].mean().reset_index()
    lowest_zone = avg_by_temp_zone.loc[avg_by_temp_zone['consumption_kwh'].idxmin()]
    st.markdown("### 🧘 Optimal Comfort Zone")
    st.write(f"You use the least energy when it's **{lowest_zone['temperature_c']}** — this is your most efficient weather range.")

    # --- Insight 3: Impact of Extreme Weather
    hot_days = daily_df[daily_df['temperature_c'] > 30]
    cold_days = daily_df[daily_df['temperature_c'] < 5]

    hot_avg = hot_days['consumption_kwh'].mean() if not hot_days.empty else None
    cold_avg = cold_days['consumption_kwh'].mean() if not cold_days.empty else None

    st.markdown("### 🌡️ Extreme Weather Energy Impact")
    if hot_avg:
        st.write(f"🔥 On hot days (>30°C), your average energy use is **{hot_avg:.2f} kWh**.")
    if cold_avg:
        st.write(f"❄️ On cold days (<5°C), your average energy use is **{cold_avg:.2f} kWh**.")
    if not hot_avg and not cold_avg:
        st.info("You haven’t experienced extreme temperatures in your data.")
