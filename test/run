#!/bin/sh
SOURCE_DIR=src/pedata
REPORTING_DIR=code_health
TEST_COV=$REPORTING_DIR/test_coverage
TEST_COV_XML=$TEST_COV.xml
TEST_REPORT=$REPORTING_DIR/test_report
TEST_REPORT_HTML=$TEST_REPORT.html
TEST_REPORT_XML=$REPORTING_DIR/test_report.xml

pytest --disable-warnings --junit-xml=$TEST_REPORT_XML --html=$TEST_REPORT_HTML --self-contained-html --cov=$SOURCE_DIR --cov-report=html:$TEST_COV --cov-report=xml:$TEST_COV_XML --cov-report=term-missing test
