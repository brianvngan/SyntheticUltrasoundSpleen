# Running the Streamlit UI

## Quick Start

The UI is completely self-contained and requires **no other setup**. You can show it right now!

### Option 1: Simple (Recommended)
```bash
pip install streamlit pandas numpy
streamlit run app_ui.py
```

That's it! Your browser will automatically open at `http://localhost:8501`

### Option 2: Full Setup (with all project dependencies)
```bash
pip install -r requirements.txt
streamlit run app_ui.py
```

## What You'll See

### 🏠 **Dashboard**
- Project overview with stats
- Quick start code
- Demo training progress chart
- Key features highlighting

### ⚙️ **Configuration**
- **Training tab:** Adjust epochs, learning rate, batch size, etc.
- **DoRA tab:** Configure LoRA rank, alpha, target modules
- **Dataset tab:** Choose which datasets to use
- **Model tab:** Select model and precision
- **Logging tab:** W&B configuration
- Save/Load/Reset buttons

### 📊 **Training**
- Start/pause/resume controls
- Live metrics graphs (loss, learning rate)
- Checkpoint management table
- Validation image preview

### 🖼️ **Inference**
- Text prompt input
- Advanced options (steps, guidance scale, seed)
- Negative prompt
- Generate button
- Demo image gallery

### 📚 **Datasets**
- Dataset statistics (1,245 images total)
- Breakdown by source (DeepSPV, Simulation, Roboflow)
- Data augmentation preview
- Preprocessing settings

### ℹ️ **About**
- Project information
- API overview (Python, CLI, REST)
- Feature checklist
- Documentation links
- Requirements met checklist

## Features

✨ **Interactive Controls:**
- Sliders for numeric parameters
- Text inputs for prompts and paths
- Multi-select for datasets and modules
- Tabs for organized content
- Styled info/success/warning boxes

✨ **Demo Data:**
- Simulated training progress graphs
- Placeholder images
- Sample checkpoint list
- Generated image gallery

✨ **Beautiful UI:**
- Dark-friendly color scheme
- Custom CSS styling
- Responsive layout with columns
- Professional card design

## No Setup Required!

Unlike the full project, this UI:
- ✅ Doesn't need datasets to be loaded
- ✅ Doesn't need GPU
- ✅ Doesn't need to train anything
- ✅ Works standalone immediately
- ✅ Shows exactly what the real system looks like

## Keyboard Shortcuts

- `c` - Clear cache (helpful if you modify the code)
- `r` - Rerun script
- `q` - Stop Streamlit

## Stopping the Server

Press `Ctrl+C` in the terminal where you ran `streamlit run app_ui.py`

## Customization

You can easily customize the UI by editing `app_ui.py`:

```python
# Change the title
st.set_page_config(page_title="Your Custom Title", ...)

# Add new sections
if page == "🆕 New Page":
    st.subheader("Your content here")

# Modify colors/styling in the CSS block at the top
```

## Troubleshooting

**"streamlit: command not found"**
```bash
pip install streamlit
```

**"ModuleNotFoundError: No module named 'pandas'"**
```bash
pip install pandas numpy
```

**Port 8501 already in use:**
```bash
streamlit run app_ui.py --server.port 8502
```

---

**That's it!** You now have a beautiful, functional UI to show right away. 🎉
