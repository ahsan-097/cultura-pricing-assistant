#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[1]:


import streamlit as st
import joblib
import pandas as pd

if "price" not in st.session_state:
    st.session_state.price = None

model = joblib.load("model.pkl")

st.title("CULTURA PRICING ASSISTANT")

st.subheader("Event Details")

event_type = st.selectbox(
    "Event Type",
    ["pilates", "sound bath", "journaling", 
     "healing circle", "meditation", "mindfulness", 
     "yoga sculpt", "community care", "breathwork", "mental health"]
)
region = st.selectbox(
    "Region",
    ["chicago", "austin", "new jersey", "salt lake city"]
)

duration_hours = st.slider("Duration (hours)", 0.75, 8.0, 2.0, 0.25)
event_intensity = st.slider("Event Intensity", 1, 3, 2)
community_accessibility = st.slider("Community Accessibility", 1, 3, 2)
trend_strength = st.slider("Trend Strength", 0, 100, 50)

col_map = {
    "chicago": 1.00,
    "austin": 0.95,
    "new jersey": 1.05,
    "salt lake city": 0.90
}

if st.button("Get Pricing Recommendation"):

    X = pd.DataFrame([{
        "duration_hours": duration_hours,
        "event_intensity": event_intensity,
        "community_accessibility": community_accessibility,
        "col_multiplier": col_map[region],
        "trend_strength": trend_strength
    }])

    st.session_state.price = model.predict(X)[0]
    price = st.session_state.price

    st.success(f"💵 Suggested Ticket Price: ${price:.2f}")

    # ---- pricing strategies ----
    low = round(price * 0.75)
    high = round(price * 1.25)

    st.markdown("### Alternate Pricing Options")
    st.write(f"• Sliding scale: ${low} – ${high}")
    st.write(f"• Tiered: ${round(price*0.9)} / ${round(price)} / ${round(price*1.2)}")

    if community_accessibility == 3:
        st.write("• Recommended: Donation-based community pricing")

    st.info("Aquí tienes tu recommendation, hermana 💜")
    


# In[3]:




# In[4]:


from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(price):
    doc = SimpleDocTemplate("pricing_report.pdf")
    styles = getSampleStyleSheet()
    content = [
        Paragraph(f"Suggested Price: ${price:.2f}", styles["Heading2"]),
        Paragraph("Cultura Pricing Assistant Report", styles["Heading1"]),
        Paragraph(f"Region: {region}", styles["BodyText"]),
        Paragraph(f"Duration: {duration_hours} hours", styles["BodyText"]),
        Paragraph(f"Event Intensity: {event_intensity}", styles["BodyText"]),
        Paragraph(f"Community Accessibility: {community_accessibility}", styles["BodyText"]),
        Paragraph(f"Trend Strength: {trend_strength}", styles["BodyText"]),
        Paragraph(" ", styles["BodyText"]),
        Paragraph(f"Suggested Price: ${price:.2f}", styles["Heading2"]),
        Paragraph(f"Sliding Scale Range: ${low} – ${high}", styles["BodyText"]),
        Paragraph("This recommendation is based on regional cost adjustments, demand signals, "
            "and event characteristics.", styles["BodyText"])
    ]
    doc.build(content)


# In[5]:


if st.session_state.price is not None:
    if st.button("Download PDF"):
        generate_pdf(st.session_state.price,
                     duration_hours, 
                     event_intensity,
                     community_accessibility,
                     trend_strength,
                     region)
        
        st.download_button(
            "Download Report",
            open("pricing_report.pdf", "rb"),
            file_name="pricing_report.pdf"
        )

# In[ ]:




