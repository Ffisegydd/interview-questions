import textwrap
import streamlit as st

from streamlit_utils.state import session_state

from consts import Level
from sub_topic_generation import generate_sub_topics


@session_state
class State:
    topic: str = None
    level: Level = None
    sub_topics: str = ""


state_expander = st.expander("State")

if __name__ == "__main__":
    st.title("Interview Questions Generator")

    st.markdown(
        textwrap.dedent(
            """
            Instructions:
            1. Enter a topic into the below text box.
            2. Select the level of the questions you want to generate.
            3. Press the 'Generate Sub-topics' button.
            """
        )
    )

    with st.expander("State"):
        state_placeholder = st.empty()
        if st.button("Write State"):
            state_placeholder.empty()
            state_placeholder.write(State)

    with st.form(key="candidate-level-form"):
        level = st.selectbox(
            "Select the candidate level of the questions",
            [level.value for level in Level],
        )

        submitted = st.form_submit_button("Select Level")
        if submitted:
            State.level = Level(level)
            st.write(f"Selected level: {level}")

    with st.form(key="interview-questions-form"):
        topic = st.text_input("Enter a topic")

        submitted = st.form_submit_button("Generate Sub-topics")
        if submitted:
            State.topic = topic
            st.write(f"Generating sub-topics for the topic: {topic}...")

            sub_topics = generate_sub_topics(topic=State.topic)

            State.sub_topics = "\n".join(sub_topics)

    txt = st.text_area(
        key="sub_topics",
        label="Sub-topics to generate questions for (one sub-topic per line)",
    )
