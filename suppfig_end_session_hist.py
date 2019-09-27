"""
HISTOGRAM OF SESSION END STATUSES DURING TRAINING
Miles  Wells, UCL, 2019
"""

import datajoint as dj
from paper_behavior_functions import figpath, query_sessions
import matplotlib.pyplot as plt
import os

# Set default figure size.
plt.rcParams['figure.figsize'] = (8, 5)
save_path = figpath()  # Our figure save path

endcriteria = dj.create_virtual_module('SessionEndCriteria', 'group_shared_end_criteria')
sessions = query_sessions()  # Query all sessions
df = (endcriteria.SessionEndCriteria & sessions).fetch(format='frame')  # Fetch data

# Convert statuses to numerical
ids = {k[0]: v for v,k in enumerate(endcriteria.EndCriteria.fetch())}
df['end_status_id'] = df['end_status'].map(ids)

ax = df['end_status_id'].value_counts().plot.bar()  # Plot the number of each

# Format axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)  # Remove ticks

# Add a horizontal grid
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)
ax.yaxis.set_ticklabels([])
ax.xaxis.set_ticklabels([id.replace('_',' ') for id in ids.keys()])
ax.xaxis.set_tick_params(rotation=0)

# Add labels and a title. Note the use of `labelpad` and `pad` to add some
# extra space between the text and the tick labels.
ax.set_xlabel('Criterion', labelpad=15, color='#333333')

total = df.shape[0]
for bar in ax.patches:
    pct = (bar.get_height() / total) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        '%.1f%%' % pct,
        horizontalalignment='center')


plt.tight_layout()
plt.gcf().savefig(os.path.join(save_path, "suppfig_end_status_histogram.png"))
