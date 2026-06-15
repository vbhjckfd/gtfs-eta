"""
Apply (or print) the R2 bucket lifecycle policy.

The collector Worker archives the served feed under `predictions/` on every
5-min cron fire (see worker/worker.js, archiveFeed).  These objects are a
bounded sample for quality scoring, not source of truth, so they must expire —
otherwise the prefix grows by ~288 objects/day forever.  This script installs a
14-day expiry on `predictions/` while leaving every other prefix (Bronze raw/,
Silver positions/, static/, feed/, quality/) untouched and permanent.

R2 implements the S3 PutBucketLifecycleConfiguration API.  Run:

    python scripts/set_r2_lifecycle.py          # apply
    python scripts/set_r2_lifecycle.py --show    # print the live policy, no change

Idempotent: re-applying the same rule set is a no-op in effect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ.get("R2_BUCKET", "gtfs-lviv")

PREDICTIONS_EXPIRY_DAYS = 14

# Only `predictions/` is governed here.  raw/ is the immutable replay source and
# must never appear in an expiry rule (see docs/collector_rules.md §5).
LIFECYCLE = {
    "Rules": [
        {
            "ID": "expire-predictions-archive",
            "Status": "Enabled",
            "Filter": {"Prefix": "predictions/"},
            "Expiration": {"Days": PREDICTIONS_EXPIRY_DAYS},
        }
    ]
}


def _make_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the bucket's current lifecycle policy and exit (no change)",
    )
    args = parser.parse_args()

    client = _make_client()

    if args.show:
        try:
            resp = client.get_bucket_lifecycle_configuration(Bucket=R2_BUCKET)
            print(json.dumps(resp.get("Rules", []), indent=2))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchLifecycleConfiguration":
                print("(no lifecycle policy set)")
            else:
                raise
        return 0

    client.put_bucket_lifecycle_configuration(
        Bucket=R2_BUCKET,
        LifecycleConfiguration=LIFECYCLE,
    )
    print(
        f"Applied lifecycle to {R2_BUCKET}: "
        f"expire predictions/ after {PREDICTIONS_EXPIRY_DAYS} days."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
