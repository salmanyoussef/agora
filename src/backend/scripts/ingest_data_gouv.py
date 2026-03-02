from __future__ import annotations

import argparse
import logging

from app.pipelines.retrieval import ingest_data_gouv


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="single_page", choices=["single_page", "all_pages"])
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--q", default=None)
    ap.add_argument("--hard-limit", type=int, default=None)
    args = ap.parse_args()

    logger.info(
        "CLI ingest_data_gouv starting: mode=%s, page=%d, page_size=%d, q=%s, hard_limit=%s",
        args.mode,
        args.page,
        args.page_size,
        args.q,
        args.hard_limit,
    )

    n = ingest_data_gouv(
        mode=args.mode,
        page=args.page,
        page_size=args.page_size,
        q=args.q,
        hard_limit=args.hard_limit,
    )

    logger.info("CLI ingest_data_gouv completed: ingested=%d", n)
    print(f"Ingested {n} datasets into Weaviate.")


if __name__ == "__main__":
    main()
