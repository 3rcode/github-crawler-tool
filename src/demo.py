from crawler.make_data import MakeData
from crawler.preprocess import PreProcess

vercel_swr_crawler = MakeData("vercel", "swr")

vercel_swr_crawler.crawl_commits_after_clone()
