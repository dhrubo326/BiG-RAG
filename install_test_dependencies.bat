@echo off
echo ========================================
echo BiG-RAG Test Dependencies Installer
echo ========================================
echo.

echo Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing BiG-RAG test dependencies...
pip install -r requirements_test.txt

echo.
echo Downloading NLP models...
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo   1. Ensure openai_api_key.txt contains your API key
echo   2. Run: python test_build_graph.py
echo   3. Run: python test_retrieval.py
echo   4. Run: python test_end_to_end.py
echo.
pause
