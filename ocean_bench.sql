/*
 Navicat Premium Data Transfer

 Source Server         : postgres
 Source Server Type    : PostgreSQL
 Source Server Version : 110005
 Source Host           : localhost:5432
 Source Catalog        : postgres
 Source Schema         : public

 Target Server Type    : PostgreSQL
 Target Server Version : 110005
 File Encoding         : 65001

 Date: 15/11/2025 17:33:56
*/


-- ----------------------------
-- Table structure for ocean_bench
-- ----------------------------
DROP TABLE IF EXISTS "public"."ocean_bench";
CREATE TABLE "public"."ocean_bench" (
  "id" SERIAL PRIMARY KEY NOT NULL,
  "task_type" varchar(255) COLLATE "pg_catalog"."default",
  "input" varchar(65535) COLLATE "pg_catalog"."default",
  "output" varchar(65535) COLLATE "pg_catalog"."default",
  "test_result" json,
  "data_type" varchar(255) COLLATE "pg_catalog"."default"
)
;

-- ----------------------------
-- Primary Key structure for table ocean_bench
-- ----------------------------
ALTER TABLE "public"."ocean_bench" ADD CONSTRAINT "ocean_bench_pkey" PRIMARY KEY ("id");
