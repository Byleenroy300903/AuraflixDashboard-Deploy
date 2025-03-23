import requests

API_KEY = ""
CX_ID = ""
query = "MS Dhoni"


YOUTUBE_API_KEY = ""

def get_youtube_engagement(public_figure_name):
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={public_figure_name}&maxResults=10&type=video&key={YOUTUBE_API_KEY}"
    search_response = requests.get(search_url).json()

    video_ids = []
    if 'items' in search_response:
        for item in search_response['items']:
            if 'id' in item and 'videoId' in item['id']:
                video_ids.append(item['id']['videoId'])

    if not video_ids:
        return {"average_views": 0, "average_likes": 0, "average_comments": 0, "total_videos": 0}

    stats_url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={','.join(video_ids)}&key={YOUTUBE_API_KEY}"
    stats_response = requests.get(stats_url).json()

    total_views = 0
    total_likes = 0
    total_comments = 0
    total_videos = len(video_ids)

    for video in stats_response.get("items", []):
        stats = video.get("statistics", {})
        total_views += int(stats.get("viewCount", 0))
        total_likes += int(stats.get("likeCount", 0))
        total_comments += int(stats.get("commentCount", 0))

    return {
        "average_views": total_views // total_videos if total_videos else 0,
        "average_likes": total_likes // total_videos if total_videos else 0,
        "average_comments": total_comments // total_videos if total_videos else 0,
        "total_videos": total_videos
    }

# **🔬 Test the function**
name = "MS Dhoni"
youtube_data = get_youtube_engagement(name)
print(f"YouTube Engagement for {name}: {youtube_data}")