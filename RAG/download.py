import nltk
import ssl

# This block is often necessary on Windows to avoid SSL certificate errors
# when downloading NLTK data.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Attempting to download NLTK 'punkt_tab' resource...")
try:
    nltk.download('punkt_tab')
    print("NLTK 'punkt_tab' resource downloaded successfully!")
except Exception as e:
    print(f"Failed to download 'punkt_tab'. Error: {e}")
    print("Trying to download 'punkt' as a fallback, as it often contains the necessary components.")
    try:
        nltk.download('punkt')
        print("NLTK 'punkt' resource downloaded successfully as a fallback.")
    except Exception as e_fallback:
        print(f"Failed to download 'punkt' as well. Please check your internet connection or try `nltk.download('all')`.")
        print(f"Fallback error: {e_fallback}")