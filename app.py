# import streamlit as st
# import os
# import plotly.express as px
# from src.inference_video import process_video

# st.title("🚗 Driver Drowsiness Detection Dashboard")
# st.write("Upload a video and detect alert/drowsy periods using your trained CNN + LSTM model.")

# uploaded = st.file_uploader("Upload Video (.avi or .mp4)", type=["avi", "mp4"])

# if uploaded:
#     # save uploaded file
#     input_path = os.path.join("temp", uploaded.name)
#     os.makedirs("temp", exist_ok=True)

#     with open(input_path, "wb") as f:
#         f.write(uploaded.getbuffer())

#     st.video(uploaded)

#     st.write("### 🔄 Processing video ... please wait")
#     output_path = os.path.join("temp", "output.avi")

#     timeline = process_video(input_path, output_path)

#     st.success("Processing complete!")

#     st.write("### 🎬 Output Video")
#     st.video(output_path)

#     # compute stats
#     alert_count = timeline.count("ALERT")
#     drowsy_count = timeline.count("DROWSY")

#     st.write("### 📊 Drowsiness Summary")
#     st.write(f"**Alert Frames:** {alert_count}")
#     st.write(f"**Drowsy Frames:** {drowsy_count}")

#     # generate timeline graph
#     numeric_timeline = [1 if t=="DROWSY" else 0 for t in timeline]
#     fig = px.line(numeric_timeline, title="Drowsiness Timeline (1 = Drowsy, 0 = Alert)")
#     fig.update_layout(xaxis_title="Frame", yaxis_title="State")
#     st.plotly_chart(fig)

#     # download button
#     with open(output_path, "rb") as f:
#         st.download_button(
#             "📥 Download Processed Video",
#             f,
#             file_name="processed_output.avi"
#         )

import streamlit as st
import os
import plotly.express as px
from src.inference_video import process_video

st.title("🚗 Driver Drowsiness Detection Dashboard")
st.write("Upload a video and detect alert/drowsy periods using your trained CNN + LSTM model.")

uploaded = st.file_uploader("Upload Video (.avi or .mp4)", type=["avi", "mp4"])

if uploaded:
    # save uploaded file
    input_path = os.path.join("temp", uploaded.name)
    os.makedirs("temp", exist_ok=True)

    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # show uploaded preview (streamlit handles the uploaded file directly)
    st.video(input_path)

    st.write("### 🔄 Processing video ... please wait")
    output_mp4 = os.path.join("temp", "output.mp4")

    timeline, mp4_path = process_video(input_path, output_mp4)

    st.success("Processing complete!")

    st.write("### 🎬 Output Video")
    # show the mp4 that was written
    st.video(mp4_path)

    # compute stats
    alert_count = timeline.count("ALERT")
    drowsy_count = timeline.count("DROWSY")

    st.write("### 📊 Drowsiness Summary")
    st.write(f"**Alert Frames:** {alert_count}")
    st.write(f"**Drowsy Frames:** {drowsy_count}")

    # generate timeline graph
    numeric_timeline = [1 if t == "DROWSY" else 0 for t in timeline]
    fig = px.line(numeric_timeline, title="Drowsiness Timeline (1 = Drowsy, 0 = Alert)")
    fig.update_layout(xaxis_title="Frame", yaxis_title="State")
    st.plotly_chart(fig)

    # download button
    with open(mp4_path, "rb") as f:
        st.download_button(
            "📥 Download Processed Video",
            f,
            file_name="processed_output.mp4"
        )
