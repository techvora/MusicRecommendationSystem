from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import pickle

app = FastAPI()

# Load data and similarity matrix
df = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.post("/recommend/")
async def recommend_music(request: Request):
    data = await request.json()
    song_name = data.get("song_name")  # Update to match the name used in the HTML form
    print(song_name)
    # Check if df is not empty
    if not df.empty:
        # Filter DataFrame to rows where 'song' column matches the input
        filtered_df = df[df['song'] == song_name]
        if not filtered_df.empty:
            # Retrieve the index of the first matching row
            idx = filtered_df.index[0]
            # Compute recommendations based on the index
            distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
            songs = [df.iloc[i[0]]['song'] for i in distances[1:21]]
            return songs
        else:
            return [f"No songs found matching '{song_name}'."]
    else:
        return ["DataFrame is empty."]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
