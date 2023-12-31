from typing import List, Optional, Tuple

from bs4 import BeautifulSoup
from markdown import markdown


class PreProcess:
    def __init__(self):
        pass

    def split_commit(self, commit: str) -> Optional[Tuple[str, str]]:
        """Split commit into commit message (the first line) and follow by commit description

        Args:
            commit (str): Full commit message

        Returns:
            Commit message title
            Commit message description
        """

        try:
            # Convert markdown into html
            html = markdown(commit)
            soup = BeautifulSoup(html, "html.parser")
            lines = [p.text.strip() for p in soup.find_all("p")]
            message = lines[0]
            description = "<.> ".join(lines[1:])
            return message, description
        except:
            return None, None

    def split_release_note_sentences(self, release_note: str) -> List[str]:
        """Split release note sentences from raw release note

        Args:
            release_note (str): Raw release note

        Returns:
            List of release note sentence
        """

        try:
            html = markdown(release_note)
            soup = BeautifulSoup(html, "html.parser")
            sentences_li = [li.text.strip() for li in soup.find_all("li")]
            sentences_p = [p.text.strip().split("\n") for p in soup.find_all("p")]
            sentences_p = [
                sentence[i] for sentence in sentences_p for i in range(len(sentence))
            ]
            sentences = [*sentences_li, *sentences_p]

            return sentences
        except:
            return []
