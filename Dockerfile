FROM sport-base:v1
LABEL org.opencontainers.image.title="Pingpang multiview landing detection"
LABEL org.opencontainers.image.version="2026-11-15-frozen86"
WORKDIR /participant
COPY participant/ /participant/
COPY selftest.sh /selftest.sh
RUN chmod +x /participant/run_all.sh /participant/pingpang/run.sh /participant/basketball/run.sh \
    && chmod +x /selftest.sh \
    && cd /participant/pingpang && python3 integrity.py --verify \
    && cd /participant/basketball && python3 integrity.py --verify
ENTRYPOINT ["/participant/run_all.sh"]
CMD []
