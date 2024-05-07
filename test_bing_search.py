import requests

def search_bing(query, api_key):
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        search_results = response.json()
        return search_results
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    query = "Is it true that Michael Jordan played for the LA Lakers?"
    api_key = ""  # Replace with your Bing API key
    search_results = search_bing(query, api_key)
    if search_results:
        res = []
        for i in search_results["webPages"]["value"]:
            res.append({"name": i["name"],
            "url": i["url"],
            "snippet": i["snippet"]})
        print(res)
    else:
        print("Failed to retrieve search results.")

if __name__ == "__main__":
    main()
