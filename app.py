import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="Facebook Ego Networks", layout="wide")

DATA_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
EGO_IDS = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]


@st.cache_data
def load_data():
    edges = pd.read_csv(DATA_URL, sep=" ", names=["source", "target"], compression="gzip")

    # Degree = number of friends per user
    all_nodes = pd.concat([edges["source"], edges["target"]])
    nodes = all_nodes.value_counts().reset_index()
    nodes.columns = ["node_id", "degree"]
    nodes["role"] = np.where(nodes["node_id"].isin(EGO_IDS), "Ego (surveyed)", "Alter (friend)")

    # Assign each user to an ego network (the ego they are connected to)
    ego_edges = edges[edges["source"].isin(EGO_IDS) | edges["target"].isin(EGO_IDS)].copy()
    ego_edges["ego"] = np.where(ego_edges["source"].isin(EGO_IDS), ego_edges["source"], ego_edges["target"])
    ego_edges["member"] = np.where(ego_edges["source"].isin(EGO_IDS), ego_edges["target"], ego_edges["source"])
    member_map = (
        ego_edges.groupby("member")["ego"].min().reset_index()
        .rename(columns={"member": "node_id"})
    )
    ego_self = pd.DataFrame({"node_id": EGO_IDS, "ego": EGO_IDS})
    member_map = pd.concat([ego_self, member_map]).drop_duplicates("node_id")

    nodes = nodes.merge(member_map, on="node_id", how="left")
    nodes["ego"] = nodes["ego"].fillna(-1).astype(int)
    nodes["ego_label"] = np.where(nodes["ego"] == -1, "No ego link", "Ego " + nodes["ego"].astype(str))
    return nodes


nodes = load_data()

st.title("Who Has the Most Friends? Exploring the ego-Facebook Network")
st.write("Data: [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html). "
         "Each user is a node, and 'degree' is their number of friends.")

# Plot 1: degree distribution (CCDF, log-log)
st.header("Plot 1: How rare are highly connected users?")

ego_choice = st.selectbox("Show users from:", ["All networks"] + sorted(nodes["ego_label"].unique()))
sub = nodes if ego_choice == "All networks" else nodes[nodes["ego_label"] == ego_choice]

ccdf = (
    sub.groupby("degree").size().sort_index(ascending=False).cumsum()
    .rename("users_at_least").reset_index().sort_values("degree")
)
ccdf["share"] = ccdf["users_at_least"] / len(sub)

plot1 = (
    alt.Chart(ccdf)
    .mark_line(point=True, color="#4c78a8")
    .encode(
        x=alt.X("degree:Q", scale=alt.Scale(type="log"), title="Number of friends (degree, log scale)"),
        y=alt.Y("share:Q", scale=alt.Scale(type="log"), title="Share of users with at least this many friends (log scale)",
                axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("degree:Q", title="Friends"),
                 alt.Tooltip("users_at_least:Q", title="Users with at least this many"),
                 alt.Tooltip("share:Q", title="Share", format=".2%")],
    )
    .properties(height=400)
)
st.altair_chart(plot1, use_container_width=True)

st.subheader("Write-up: Plot 1")
# describes what features you are highlighting with your viz,
# the design choices you made (choices of scales, marks, colors, etc) to visualize your data 
# appropriately,
# what you would like to change if you had more time
st.write(
    "**Features highlighted:** I show that most users have few friends, and a small number have very many." 
    " Also, filtering by ego network shows whether every network has this same shape. (They mostly do.)"
    " The line drops steeply on the right. This shows that users that act as hubs (those with a high degree) are rare.\n\n"
    "**Design choices:** I used a continuous line with points to show how the share of users changes as friend count grows."
    " I used a log scale on both axes because friend counts range from 1 to over 1,000, so a linear scale would squash most of the data into a corner."
    " The y-axis is a percent share, so networks of different sizes can be compared fairly."
    " I chose to use only a single blue color, because there`s only one variable and no categories to separate."
    " The tooltip gives exact numbers, and the dropdown lets viewers explore one network at a time.\n\n"
    "**What I would change:** If I had more time, I would overlay all ten ego networks as separate lines with a legend, adjusting color"
    " and opacity as needed. I would also add a main, opaque reference line for the average degree and label the axes more simply, since “log scale” may confuse some readers.."
)

# Plot 2: top hubs
st.header("Plot 2: The biggest hubs")

top_n = st.slider("Number of top users to show", 5, 50, 15)
role_filter = st.multiselect("User role", ["Ego (surveyed)", "Alter (friend)"],
                             default=["Ego (surveyed)", "Alter (friend)"])
top = nodes[nodes["role"].isin(role_filter)].nlargest(top_n, "degree").copy()
top["user"] = "User " + top["node_id"].astype(str)

plot2 = (
    alt.Chart(top)
    .mark_bar()
    .encode(
        y=alt.Y("user:N", sort="-x", title="User"),
        x=alt.X("degree:Q", title="Number of friends"),
        color=alt.Color("role:N", title="Role",
                        scale=alt.Scale(domain=["Alter (friend)", "Ego (surveyed)"],
                                        range=["#4c78a8", "#e45756"])),
        tooltip=[alt.Tooltip("user:N", title="User"),
                 alt.Tooltip("degree:Q", title="Friends"),
                 alt.Tooltip("ego_label:N", title="Ego network")],
    )
    .properties(height=max(200, top_n * 22))
)
st.altair_chart(plot2, use_container_width=True)

st.subheader("Write-up: Plot 2")
st.write("**Features highlighted:** I show which individual users have the most friends,"
         " whether the biggest hubs are egos (surveyed users) or alters (friends),"
         " and how big the gap is between the top user and the rest.\n\n"
          "**Design choices:** I used horizontal bars sorted by value because it's easy to compare lengths,"
          " and the user labels stay readable. I used a linear x-axis, since this chart is about direct comparison. "
          "I colored the users by role, so red for egos, blue for alters. Red and blue are easy to tell apart. "
          "The slider controls how many users to show, so the viewer can go from a quick top 5 to a longer list without clutter. "
          "The role filter lets viewers compare egos and alters separately.\n\n"
          "**What I would change:** I would show which ego network each hub belongs to by color or grouping and replace anonymous user IDs " 
          "with a more identifiable label. I could also add a click-to-highlight selection inside the chart.")

# Overlap with HW5 
st.header("Overlap with Homework #5")
st.write(
    "I used the same dataset (SNAP ego-Facebook) and reused the data-processing steps from Homework #5 "
    "(degree counts and ego-network assignment). "
    "Both plots are new: HW5 used a histogram of degree and a bar chart of ego-network sizes; "
    "this app uses a log-log distribution line chart and a top-users bar chart with Streamlit widgets. "
)
