$ErrorActionPreference = "Stop"
docker compose -f docker-compose.publication.yml up --build publication-test
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
