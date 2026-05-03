#!/usr/bin/env bash
set -euo pipefail

PROFILE="personal"
ZIP_NAME="deploy.zip"
BUILD_DIR="build"
REQ_FILE="requirements.txt"
S3_BUCKET="${S3_BUCKET:-}"
REGION="eu-central-1"
LAMBDA_NAME="business_simple_ai_bot_prod"
DEFAULT_S3_BUCKET="business-simple-ai-bot-prod-artifacts"

if [[ -z "$S3_BUCKET" ]]; then
  S3_BUCKET="$DEFAULT_S3_BUCKET"
fi

echo "==> Cleaning build artifacts"
rm -rf "$BUILD_DIR" "$ZIP_NAME"
mkdir -p "$BUILD_DIR"

echo "==> Installing dependencies"
python3 -m pip install -r "$REQ_FILE" \
  --platform manylinux2014_x86_64 \
  --python-version 312 \
  --only-binary=:all: \
  --implementation cp \
  --target "$BUILD_DIR"

echo "==> Copying source files"
cp -r app.py src "$BUILD_DIR"/

echo "==> Creating deployment zip"
(
  cd "$BUILD_DIR"
  zip -r "../$ZIP_NAME" .
)

S3_KEY="${LAMBDA_NAME}/deploy-$(date +%Y%m%d%H%M%S).zip"

echo "==> Uploading deployment zip to s3://$S3_BUCKET/$S3_KEY"
aws s3 cp "$ZIP_NAME" "s3://$S3_BUCKET/$S3_KEY" \
  --region "$REGION" \
  --profile "$PROFILE" \
  --no-cli-pager

echo "==> Updating Lambda $LAMBDA_NAME from S3"
aws lambda update-function-code \
  --function-name "$LAMBDA_NAME" \
  --s3-bucket "$S3_BUCKET" \
  --s3-key "$S3_KEY" \
  --region "$REGION" \
  --profile "$PROFILE" \
  --no-cli-pager \
  --query 'FunctionArn' \
  --output text

echo "✅ Deployment complete."
