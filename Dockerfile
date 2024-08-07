FROM dockerhub.apps.cp.meteoswiss.ch/mch/python/builder AS builder
ARG VERSION
LABEL ch.meteoswiss.project=probtest-${VERSION}

COPY poetry.lock pyproject.toml /src/app-root/

WORKDIR /src/app-root

RUN poetry export -o requirements.txt --without-hashes \
  && poetry export --with dev -o requirements_dev.txt --without-hashes


FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.11:latest-slim AS base
ARG VERSION
LABEL ch.meteoswiss.project=probtest-${VERSION}

COPY --from=builder /src/app-root/requirements.txt /src/app-root/requirements.txt

WORKDIR /src/app-root

RUN pip install -r requirements.txt --no-cache-dir --no-deps --root-user-action=ignore

COPY probtest /src/app-root/probtest

FROM base AS tester
ARG VERSION
LABEL ch.meteoswiss.project=probtest-${VERSION}

COPY --from=builder /src/app-root/requirements_dev.txt /src/app-root/requirements_dev.txt
RUN pip install -r /src/app-root/requirements_dev.txt --no-cache-dir --no-deps --root-user-action=ignore

COPY pyproject.toml /src/app-root/
COPY test /src/app-root/test

FROM base AS runner
ARG VERSION
LABEL ch.meteoswiss.project=probtest-${VERSION}

ENV VERSION=$VERSION

# For running outside of OpenShift, we want to make sure that the container is run without root privileges
# uid 1001 is defined in the base-container-images for this purpose
USER 1001

ENTRYPOINT ["python", "-m", "probtest"]
CMD []
