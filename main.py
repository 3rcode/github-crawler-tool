from crawl import crawl_releases, crawl_compare_commit

if __name__ == "__main__":
    releases = crawl_releases("spring-projects", "react")
    compare_commits = crawl_compare_commit("facebook", "react", releases)
    for compare_commit in compare_commits:
        if len(compare_commit[2]) < 10:
            del compare_commit
    compare_commits = list(filter(lambda x: len(x[2]) >= 10, compare_commits))
    with open("compare_commits.txt", "w", encoding="utf-8") as f:
        for compare_commit in compare_commits:
            f.write(f"{compare_commit[0]} {compare_commit[1]} {compare_commit[2]}")

