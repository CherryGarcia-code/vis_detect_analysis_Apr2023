# Instructions for Copilot

You are assisting with analysis and visualization of cortico-basal ganglia photometry recordings from a visual change-detection task in mice.

## Tasks to Prioritize

- **Data Wrangling:**  
  - Create Python data classes to represent sessions, trials, photometry data, and behavioral events using session schema.
  - Ensure classes are flexible for batch processing and easy access to trial-level data.

- **Analysis:**  
  - Align neural data (photometry signal) with behavioral events (trial outcomes, stimulus changes).
  - Compute and visualize peri-event time histograms (PETHs) for different trial outcomes.
  - Group events by response patterns (e.g., clustering, PCA).
  - Track units across sessions if possible.

- **Dimensionality Reduction:**
  - Apply dimensionality reduction techniques (e.g., PCA, t-SNE, UMAP, and coding direction analyses) to population activity.
  - Use these methods to visualize and interpret high-dimensional data, identify patterns, and relate trajectories to behavioral events.


- **Visualization:**  
  - Plot trial-by-trial neural and behavioral data (heatmaps, raster plots, summary stats).
  - Visualize learning curves and changes in neural activity over training.

## Coding Preferences

- Use modern Python (3.9+), pandas, numpy, matplotlib, seaborn, and scikit-learn.
- Avoid deprecated functions.
- Write clear, well-commented code.
- Where possible, use type hints and docstrings.

## Data Reference


- If unsure about a field, ask for clarification or suggest a reasonable default.

## Research Focus

- Prioritize analyses that address the research questions listed in the `README (2).md`.
- Suggest additional analyses or visualizations if they could provide new insights.
