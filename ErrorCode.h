// ErrorCode.h
//
// �����루Error Codes��
// ����˵����������ϵͳ��ͨ�õĴ������״̬�롣������ ʹ�ø�������ȷִ��ʹ�� 0��
//           ����ķǴ���״̬ʹ���������û��͸��㷨��
//           �Ը����Լ�����Ҫ�������Ĵ������״̬�롣
//
// �޶���ʷ��
//     ��ʼ�汾��

#ifndef __ERRORCODE_H__
#define __ERRORCODE_H__

// �꣺NO_ERROR���޴���
// ����ʾ������ȷִ�У�û�д�������
#define NO_ERROR       0

// �꣺INVALID_DATA����Ч���ݣ�
// ����ʾ�����а�������Ч�����ݡ�
#define INVALID_DATA  -1

// �꣺NULL_POINTER����ָ�룩
// ����ʾ����Ϊ NULL �ı��������������ĳ����� NULL ֵ��
#define NULL_POINTER  -2

// �꣺OVERFLOW�����������
// ����ʾ������ĳЩ�����������������а����˳������
#define OP_OVERFLOW   -3

// �꣺NO_FILE���ļ�δ�ҵ���
// ����ʾ����δ�ҵ�ָ�����ļ���
#define NO_FILE       -4

// �꣺WRONG_FILE���ļ�����
// ����ʾ�������ļ��ĸ�ʽ�Ǵ���ġ�
#define WRONG_FILE    -5

// �꣺OUT_OF_MEM���ڴ�ľ���
// ����ʾ��ǰ��û�ж�����ڴ�֧����Ҫ���еĲ����ˡ�
#define OUT_OF_MEM    -6

// �꣺CUDA_ERROR��CUDA ����
// ����ʾ���� CUDA ���ñ����޷����������Ӧ�Ĳ�����
#define CUDA_ERROR    -7

// �꣺UNMATCH_IMG��ͼ��ߴ粻ƥ�䣩
// ����ʾ������ͼ��ߴ�Ͳ�����Ҫ���ͼ��ߴ粻ƥ�䣬�޷���һ����ɲ�����
#define UNMATCH_IMG   -8

// �꣺UMIMPLEMENT��δʵ�֣�
// ����ʾ�����õĲ�����δʵ�֣��ô��󲻻�������ύ��ί�з��Ĵ����С�
#define UNIMPLEMENT   -998

// �꣺UNKNOW_ERROR��δ֪����
// ����ʾϵͳ�ɶ϶���һ�����󣬵�������������ԭ��
#define UNKNOW_ERROR  -999


#endif