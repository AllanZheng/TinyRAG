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

 Date: 17/11/2025 11:17:58
*/


-- ----------------------------
-- Table structure for triples_1115
-- ----------------------------
DROP TABLE IF EXISTS "public"."triples_1115";
CREATE TABLE "public"."triples_1115" (
  "id" SERIAL PRIMARY KEY NOT NULL,
  "start" varchar(65535) COLLATE "pg_catalog"."default",
  "mid" varchar(65535) COLLATE "pg_catalog"."default",
  "finish" varchar(65535) COLLATE "pg_catalog"."default",
  "key" varchar(255) COLLATE "pg_catalog"."default",
  "source" varchar(255) COLLATE "pg_catalog"."default"
)
;

-- ----------------------------
-- Records of triples_1115
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table triples_1115
-- ----------------------------
ALTER TABLE "public"."triples_1115" ADD CONSTRAINT "triples_1115_pkey" PRIMARY KEY ("id");
