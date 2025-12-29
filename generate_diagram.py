import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Boxes
ax.add_patch(patches.Rectangle((1, 6), 2, 1, facecolor='lightgreen', edgecolor='black'))
ax.text(2, 6.5, 'Farmer/User\n(Start)', ha='center')

ax.add_patch(patches.Rectangle((5, 6), 2, 1, facecolor='lightblue', edgecolor='black'))
ax.text(6, 6.5, 'Streamlit\nWeb App', ha='center')

ax.add_patch(patches.Rectangle((9, 6), 2, 1, facecolor='lightcoral', edgecolor='black'))
ax.text(10, 6.5, 'Farmer Receives\nAdvice (End)', ha='center')

# Arrows
ax.arrow(3, 6.5, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
ax.arrow(7, 6.5, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Sub-processes
ax.add_patch(patches.Rectangle((1, 4), 2, 1, facecolor='yellow', edgecolor='black'))
ax.text(2, 4.5, 'Soil Data\nInput', ha='center')

ax.add_patch(patches.Rectangle((5, 4), 2, 1, facecolor='yellow', edgecolor='black'))
ax.text(6, 4.5, 'Random Forest\nModel', ha='center')

ax.add_patch(patches.Rectangle((9, 4), 2, 1, facecolor='yellow', edgecolor='black'))
ax.text(10, 4.5, 'Crop +\nSHAP', ha='center')

# Disease path
ax.add_patch(patches.Rectangle((1, 2), 2, 1, facecolor='lightcyan', edgecolor='black'))
ax.text(2, 2.5, 'Plant Image\nUpload', ha='center')

ax.add_patch(patches.Rectangle((5, 2), 2, 1, facecolor='lightcyan', edgecolor='black'))
ax.text(6, 2.5, 'CNN Model\n(DL)', ha='center')

ax.add_patch(patches.Rectangle((9, 2), 2, 1, facecolor='lightcyan', edgecolor='black'))
ax.text(10, 2.5, 'Disease +\nGrad-CAM', ha='center')

# Tips path
ax.add_patch(patches.Rectangle((1, 0), 2, 1, facecolor='lightpink', edgecolor='black'))
ax.text(2, 0.5, 'Language\nSelection', ha='center')

ax.add_patch(patches.Rectangle((5, 0), 2, 1, facecolor='lightpink', edgecolor='black'))
ax.text(6, 0.5, 'Translated\nTips (NLP)', ha='center')

ax.add_patch(patches.Rectangle((9, 0), 2, 1, facecolor='lightpink', edgecolor='black'))
ax.text(10, 0.5, 'Farming\nTips', ha='center')

# Arrows for sub-processes
for y in [4.5, 2.5, 0.5]:
    ax.arrow(3, y, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(7, y, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

plt.title('Block Diagram: AI Advisor for Farmers')
plt.savefig('block_diagram.png')
print('Block diagram saved as block_diagram.png')