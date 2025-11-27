import marimo

app = marimo.App()

@app.cell
def intro():
    return marimo.md("# Welcome to Marimo!")

@app.cell
def hello():
    return "Hello, Marimo!"

if __name__ == "__main__":
    app.run()
