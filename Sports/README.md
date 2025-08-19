# Sports

Welcome to the Sports project, this project lays out the template for sports predictions.

## Prerequisites

Before getting started, make sure your system meets these requirements:

### Required Software
* **Python:** Version 3.10 or higher - [Download Python](https://www.python.org/downloads/)
* **Visual Studio Code:** Latest version - [Download VS Code](https://code.visualstudio.com/download)

### Docker Desktop Requirements
* **Windows:**
  - Windows 10/11 64-bit: Pro, Enterprise, or Education (Build 16299 or later)
  - WSL 2 feature enabled
  - 4GB system RAM minimum
  - BIOS-level hardware virtualization enabled
  - [Download Docker Desktop for Windows](https://docs.docker.com/desktop/windows/install/)

* **macOS:**
  - macOS version 11 or newer (Intel or Apple Silicon)
  - At least 4GB of RAM
  - [Download Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)

* **Linux:**
  - 64-bit kernel and CPU support for virtualization
  - systemd init system
  - At least 4GB of RAM
  - [Download Docker Desktop for Linux](https://docs.docker.com/desktop/linux/install/)

### Fire Up Your Model with Docker!
Time to unleash your model! In your *same* terminal, run these commands:

1. **Change directory into the sports template:** `cd Sports` 
1. **Build the magic image:** `docker build -t sports-predictor .`
2. **Run your new image:** `docker run -p 8000:8000 sports-predictor`

*Keep this terminal running! It's busy making predictions!*

---

### Is It Working? Let's Find Out!

Open your favorite web browser and navigate to: `http://localhost:8000/health`

If all is well, you'll see a happy message like this:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "last_trained_at": "2025-07-25 09:33:58"
}
```

You can also see the docs at: `http://localhost:8000/docs`

Now, let's make some predictions!

1.  In Visual Studio Code, go to the top right of your terminal, click the `+` dropdown, and select `Git Bash`. This will open a *second* terminal.
2.  In this new terminal, paste and run the following command:

    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{
             "home_team": "Lions",
             "away_team": "Sharks",
             "home_team_odds_avg": 1.75,
             "away_team_odds_avg": 2.20
           }'
    ```

    Get ready for your first prediction! How exciting is that?!

---

### Craft Your Own Prediction Masterpiece!

This is where the real fun begins! You get to customize how your model thinks.

* Open `src/models.py`. This is your canvas! Update the logic in the predict method to reflect your brilliant prediction strategy.

Made changes? Awesome! Here's how to see them in action:

1.  Go back to your *first* terminal (where the model is running) and press `Ctrl + c` to stop it.
2.  Simply repeat the "Fire Up" and "Is It Working" steps above. Easy peasy!

**Congratulations, you are now a bona fide model builder!**

---

### Peek Under the Hood: Project Files Explained

Curious about what makes the Sports model tick? Here's a quick tour:

* **`main.py`**: This is the heart of your project, the starting point for your API and where your model gets called.
* **`src/api.py`**: Defines how your model talks to the world (the API structure).
* **`src/models.py`**: **This is where YOUR custom prediction logic lives!** Get creative here.
* **`src/schemas.py`**: Lays out the data structures the API expects and provides.

And the rest:

* **`.dockerignore`**: Tells Docker what files to skip. You probably won't need to touch this often.
* **`.gitignore`**: Tells Git what files to ignore. No need to edit unless you add new files you don't want tracked.
* **`LICENSE`**: The project's license.
* **`README.md`**: What you're reading right now!
* **`requirements.txt`**: Lets Docker know which Python libraries your project needs. **Important! If you add new Python libraries to your custom logic, remember to add them here!**