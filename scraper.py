"""
Reddit Post & Comment Scraper
=============================
Scrapes posts and comments from Reddit based on a search query.
Outputs data to a CSV file with columns: [id, Reddit_Post_ID, Text].

Usage:
1. Get your Reddit API credentials from https://www.reddit.com/prefs/apps
2. Fill in CLIENT_ID, CLIENT_SECRET, and USER_AGENT below.
3. Run: python reddit_scraper.py
"""

import praw
import time
import csv
import os
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# [IMPORTANT] REPLACE THESE WITH YOUR OWN REDDIT API CREDENTIALS
CREDENTIALS = {
    "CLIENT_ID": "YOUR_CLIENT_ID_HERE",
    "CLIENT_SECRET": "YOUR_CLIENT_SECRET_HERE",
    "USER_AGENT": "script:my_ai_scraper:v1.0 (by /u/YOUR_USERNAME)"
}

SEARCH_QUERY = "artificial intelligence"
POST_LIMIT = 50             # Number of new posts to fetch per run
OUTPUT_FILENAME = "ai_dataset.csv"
# ==============================================================================

def get_reddit_instance():
    """Initialize PRAW instance."""
    if CREDENTIALS["CLIENT_ID"] == "YOUR_CLIENT_ID_HERE":
        print("[Error] You must add your Reddit API credentials in the script.")
        sys.exit(1)
        
    return praw.Reddit(
        client_id=CREDENTIALS["CLIENT_ID"],
        client_secret=CREDENTIALS["CLIENT_SECRET"],
        user_agent=CREDENTIALS["USER_AGENT"]
    )

def sanitize_text(text):
    """Clean text for CSV safety."""
    if not text: return ""
    return text.replace('\n', ' ').replace('\r', ' ').replace('"', "'")

def main():
    reddit = get_reddit_instance()
    
    print(f"--- Starting Scraper ---")
    print(f"Query: '{SEARCH_QUERY}' | Limit: {POST_LIMIT} posts")
    
    file_exists = os.path.isfile(OUTPUT_FILENAME)
    
    # Determine the next sequential ID
    current_id = 0
    if file_exists:
        with open(OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 1:
                try:
                    current_id = int(rows[-1][0]) + 1
                except ValueError:
                    current_id = 1
            else:
                current_id = 1

    posts_processed = 0
    # Open file in Append mode
    with open(OUTPUT_FILENAME, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow(["id", "Reddit_Post_ID", "Text"])

        print(f"Searching for posts...")
        
        try:
            for submission in reddit.subreddit("all").search(SEARCH_QUERY, sort="new", limit=POST_LIMIT):
                
                # Save the Post Title/Body
                post_text = f"{submission.title} {submission.selftext}"
                clean_post = sanitize_text(post_text)
                
                writer.writerow([current_id, submission.id, clean_post])
                current_id += 1
                
                # Save Comments
                submission.comments.replace_more(limit=0) # Flatten comment tree
                comment_count = 0
                for comment in submission.comments.list():
                    if hasattr(comment, 'body'):
                        clean_comment = sanitize_text(comment.body)
                        writer.writerow([current_id, submission.id, clean_comment])
                        current_id += 1
                        comment_count += 1
                
                posts_processed += 1
                print(f"  > Saved Post {submission.id} + {comment_count} comments")
                time.sleep(1.1) # Respect API limits
                
        except Exception as e:
            print(f"[Error] {e}")

    print(f"\n--- Done ---")
    print(f"Processed {posts_processed} posts.")
    print(f"Data saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()
