# PTX ISA Notes

Introduced in PTX ISA version 7.8. 

# Target ISA Notes

Requires sm_90 or higher. 

# Examples

```txt
getctarank.shared::cluster.u32 d1, addr;  
getctarank.shared::cluster.u64 d2, sh + 4;  
getctarank.u64 d3, src; 
```

# 9.7.9.25 Data Movement and Conversion Instructions: Asynchronous copy

An asynchronous copy operation performs the underlying operation asynchronously in the background, thus allowing the issuing threads to perform subsequent tasks. 

An asynchronous copy operation can be a bulk operation that operates on a large amount of data, or a non-bulk operation that operates on smaller sized data. The amount of data handled by a bulk asynchronous operation must be a multiple of 16 bytes. 

An asynchronous copy operation typically includes the following sequence: 

▶ Optionally, reading from the tensormap. 

▶ Reading data from the source location(s). 

▶ Writing data to the destination location(s). 

▶ Writes being made visible to the executing thread or other threads. 

# 9.7.9.25.1 Completion Mechanisms for Asynchronous Copy Operations

A thread must explicitly wait for the completion of an asynchronous copy operation in order to access the result of the operation. Once an asynchronous copy operation is initiated, modifying the source memory location or tensor descriptor or reading from the destination memory location before the asynchronous operation completes, exhibits undefined behavior. 

This section describes two asynchronous copy operation completion mechanisms supported in PTX: Async-group mechanism and mbarrier-based mechanism. 

Asynchronous operations may be tracked by either of the completion mechanisms or both mechanisms. The tracking mechanism is instruction/instruction-variant specific. 

# 9.7.9.25.1.1 Async-group mechanism

When using the async-group completion mechanism, the issuing thread specifies a group of asynchronous operations, called async-group, using a commit operation and tracks the completion of this group using a wait operation. The thread issuing the asynchronous operation must create separate async-groups for bulk and non-bulk asynchronous operations. 

A commit operation creates a per-thread async-group containing all prior asynchronous operations tracked by async-group completion and initiated by the executing thread but none of the asynchronous operations following the commit operation. A committed asynchronous operation belongs to a single async-group. 

When an async-group completes, all the asynchronous operations belonging to that group are complete and the executing thread that initiated the asynchronous operations can read the result of the asynchronous operations. All async-groups committed by an executing thread always complete in the order in which they were committed. There is no ordering between asynchronous operations within an async-group. 

A typical pattern of using async-group as the completion mechanism is as follows: 

▶ Initiate the asynchronous operations. 

▶ Group the asynchronous operations into an async-group using a commit operation. 

▶ Wait for the completion of the async-group using the wait operation. 

▶ Once the async-group completes, access the results of all asynchronous operations in that asyncgroup. 

# 9.7.9.25.1.2 Mbarrier-based mechanism

A thread can track the completion of one or more asynchronous operations using the current phase of an mbarrier object. When the current phase of the mbarrier object is complete, it implies that all asynchronous operations tracked by this phase are complete, and all threads participating in that mbarrier object can access the result of the asynchronous operations. 

The mbarrier object to be used for tracking the completion of an asynchronous operation can be either specified along with the asynchronous operation as part of its syntax, or as a separate operation. For a bulk asynchronous operation, the mbarrier object must be specified in the asynchronous operation, whereas for non-bulk operations, it can be specified after the asynchronous operation. 

A typical pattern of using mbarrier-based completion mechanism is as follows: 

▶ Initiate the asynchronous operations. 

▶ Set up an mbarrier object to track the asynchronous operations in its current phase, either as part of the asynchronous operation or as a separate operation. 

▶ Wait for the mbarrier object to complete its current phase using mbarrier.test_wait or mbarrier.try_wait. 

▶ Once the mbarrier.test_wait or mbarrier.try_wait operation returns True, access the results of the asynchronous operations tracked by the mbarrier object. 

# 9.7.9.25.2 Async Proxy

The cp{.reduce}.async.bulk operations are performed in the asynchronous proxy (or async proxy). 

Accessing the same memory location across multiple proxies needs a cross-proxy fence. For the async proxy, fence.proxy.async should be used to synchronize memory between generic proxy and the async proxy. 

The completion of a cp{.reduce}.async.bulk operation is followed by an implicit generic-async proxy fence. So the result of the asynchronous operation is made visible to the generic proxy as soon as its completion is observed. Async-group OR mbarrier-based completion mechanism must be used to wait for the completion of the cp{.reduce}.async.bulk instructions. 

# 9.7.9.25.3 Data Movement and Conversion Instructions: Non-bulk copy

# 9.7.9.25.3.1 Data Movement and Conversion Instructions: cp.async

cp.async 

Initiates an asynchronous copy operation from one state space to another. 

# Syntax

```txt
cpasync.ca.shared{::cta}.global{.level::cache_hint{.level::prefetch_size} [dst], [src], cp-size{, src-size}{, cache-policy};  
cpasync.cgshared{::cta}.global{.level::cache_hint{.level::prefetch_size} [dst], [src], 16{, src-size}{, cache-policy};  
cpasync.ca.shared{::cta}.global{.level::cache_hint{.level::prefetch_size} [dst], [src], cp-size{, ignore-src}{, cache-policy};  
cpasync.cgshared{::cta}.global{.level::cache_hint{.level::prefetch_size} [dst], [src], 16{, ignore-src}{, cache-policy};  
.lavel::cache_hint = { .L2::cache_hint }  
.lavel::prefetch_size = { .L2::64B, .L2::128B, .L2::256B}  
cp-size = { 4, 8, 16} 
```

# Description

cp.async is a non-blocking instruction which initiates an asynchronous copy operation of data from the location specified by source address operand src to the location specified by destination address operand dst. Operand src specifies a location in the global state space and dst specifies a location in the shared state space. 

Operand cp-size is an integer constant which specifies the size of data in bytes to be copied to the destination dst. cp-size can only be 4, 8 and 16. 

Instruction cp.async allows optionally specifying a 32-bit integer operand src-size. Operand src-size represents the size of the data in bytes to be copied from src to dst and must be less than cp-size. In such case, remaining bytes in destination dst are filled with zeros. Specifying src-size larger than cp-size results in undefined behavior. 

The optional and non-immediate predicate argument ignore-src specifies whether the data from the source location src should be ignored completely. If the source data is ignored then zeros will be copied to destination dst. If the argument ignore-src is not specified then it defaults to False. 

Supported alignment requirements and addressing modes for operand src and dst are described in Addresses as Operands. 

The mandatory .async qualifier indicates that the cp instruction will initiate the memory copy operation asynchronously and control will return to the executing thread before the copy operation is complete. The executing thread can then use async-group based completion mechanism or the mbarrier based completion mechanism to wait for completion of the asynchronous copy operation. No other synchronization mechanism guarantees the completion of the asynchronous copy operations. 

There is no ordering guarantee between two cp.async operations if they are not explicitly synchronized using cp.async.wait_all or cp.async.wait_group or mbarrier instructions. 

As described in Cache Operators, the .cg qualifier indicates caching of data only at global level cache L2 and not at L1 whereas .ca qualifier indicates caching of data at all levels including L1 cache. Cache operator are treated as performance hints only. 

cp.async is treated as a weak memory operation performed in the generic proxy in the Memory Consistency Model. 

The .level::prefetch_size qualifier is a hint to fetch additional data of the specified size into the respective cache level.The sub-qualifier prefetch_size can be set to either of 64B, 128B, 256B thereby allowing the prefetch size to be 64 Bytes, 128 Bytes or 256 Bytes respectively. 

The qualifier .level::prefetch_size may only be used with .global state space and with generic addressing where the address points to .global state space. If the generic address does not fall within the address window of the global memory, then the prefetching behavior is undefined. 

The .level::prefetch_size qualifier is treated as a performance hint only. 

When the optional argument cache-policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache-policy specifies the cache eviction policy that may be used during the memory access. 

The qualifier .level::cache_hint is only supported for .global state space and for generic addressing where the address points to the .global state space. 

cache-policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. 

# PTX ISA Notes

Introduced in PTX ISA version 7.0. 

Support for .level::cache_hint and .level::prefetch_size qualifiers introduced in PTX ISA version 7.4. 

Support for ignore-src operand introduced in PTX ISA version 7.5. 

Support for sub-qualifier ::cta introduced in PTX ISA version 7.8. 

# Target ISA Notes

Requires sm_80 or higher. 

Sub-qualifier ::cta requires sm_30 or higher. 

# Examples

```txt
cpasync.ca.shared.global [shrd], [gb1 + 4], 4;  
cpasync.ca.shared::cta.global [%r0 + 8], [%r1], 8;  
cpasync.cg_shared.global [%r2], [%r3], 16;  
cpasync.cg_shared.global.L2::64B [%r2], [%r3], 16;  
cpasync.cg_shared.global.L2::128B [%r0 + 16], [%r1], 16;  
cpasync.cg_shared.global.L2::256B [%r2 + 32], [%r3], 16;  
createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 cache-policy, 0.25;  
cpasync.ca.shared.global.L2::cache_hint [%r2], [%r1], 4, cache-policy;  
cpasync.ca.shared.global [shrd], [gb1], 4, p;  
cpasync.cg_shared.global.L2::cache_hint [%r0], [%r2], 16, q, cache-policy; 
```

# 9.7.9.25.3.2 Data Movement and Conversion Instructions: cp.async.commit_group

cp.async.commit_group 

Commits all prior initiated but uncommitted cp.async instructions into a cp.async-group. 

# Syntax

```txt
cpasync.commit_group; 
```

# Description

cp.async.commit_group instruction creates a new cp.async-group per thread and batches all prior cp.async instructions initiated by the executing thread but not committed to any cp.async-group into the new cp.async-group. If there are no uncommitted cp.async instructions then cp.async. commit_group results in an empty cp.async-group. 

An executing thread can wait for the completion of all cp.async operations in a cp.async-group using cp.async.wait_group. 

There is no memory ordering guarantee provided between any two cp.async operations within the same cp.async-group. So two or more cp.async operations within a cp.async-group copying data to the same location results in undefined behavior. 

# PTX ISA Notes

Introduced in PTX ISA version 7.0. 

# Target ISA Notes

Requires sm_80 or higher. 

# Examples

```javascript
// Example 1:  
cpasync.ca.shared.global [shrd], [gb1], 4;  
cpasync.commit_group; // Marks the end of a cpasync group  
// Example 2:  
cpasync.ca.shared.global [shrd1], [gb1], 8;  
cpasync.ca.shared.global [shrd1+8], [gb1+8], 8;  
cpasync.commit_group; // Marks the end of cpasync group 1  
cpasync.ca.shared.global [shrd2], [gb12], 16;  
cpasync.cg_shared.global [shrd2+16], [gb12+16], 16;  
cpasync.commit_group; // Marks the end of cpasync group 2 
```

# 9.7.9.25.3.3 Data Movement and Conversion Instructions: cp.async.wait_group / cp.async. wait_all

cp.async.wait_group, cp.async.wait_all 

Wait for completion of prior asynchronous copy operations. 

# Syntax

```txt
cpasync.wait_group N;  
cpasync.wait_all; 
```

# Description

cp.async.wait_group instruction will cause executing thread to wait till only N or fewer of the most recent cp.async-groups are pending and all the prior cp.async-groups committed by the executing threads are complete. For example, when N is 0, the executing thread waits on all the prior cp.asyncgroups to complete. Operand N is an integer constant. 

cp.async.wait_all is equivalent to : 

```txt
cpasync.commit_group;  
cpasync.wait_group 0; 
```

An empty cp.async-group is considered to be trivially complete. 

Writes performed by cp.async operations are made visible to the executing thread only after: 

1. The completion of cp.async.wait_all or 

2. The completion of cp.async.wait_group on the cp.async-group in which the cp.async belongs to or 

3. mbarrier.test_wait returns True on an mbarrier object which is tracking the completion of the cp.async operation. 

There is no ordering between two cp.async operations that are not synchronized with cp.async. wait_all or cp.async.wait_group or mbarrier objects. 

cp.async.wait_group and cp.async.wait_all does not provide any ordering and visibility guarantees for any other memory operation apart from cp.async. 

# PTX ISA Notes

Introduced in PTX ISA version 7.0. 

# Target ISA Notes

Requires sm_80 or higher. 

# Examples

```asm
// Example of .wait_all:  
cpasync.ca.shared.global [shrd1], [gb11], 4;  
cpasync.cg_shared.global [shrd2], [gb12], 16;  
cpasync.wait_all; // waits for all prior cpasync to complete  
// Example of .wait_group :  
cpasync.ca.shared.global [shrd3], [gb13], 8;  
cpasync.commit_group; // End of group 1  
cpasync.cg_shared.global [shrd4], [gb14], 16;  
cpasync.commit_group; // End of group 2  
cpasync.cg_shared.global [shrd5], [gb15], 16;  
cpasync.commit_group; // End of group 3  
cpasync.wait_group 1; // waits for group 1 and group 2 to complete 
```

# 9.7.9.25.4 Data Movement and Conversion Instructions: Bulk copy

# 9.7.9.25.4.1 Data Movement and Conversion Instructions: cp.async.bulk

# cp.async.bulk

Initiates an asynchronous copy operation from one state space to another. 

# 9.7. Instructions

# Syntax

```ini
// global -> shared::cta
cp.async.bulk.dst.srccompletion_mechanism {.level::cache_hint} {.ignore_oob}
[dstMem], [srcMem], size{, ignoreBytesLeft, ignoreBytesRight},
→[mbar] {, cache-policy}
.dst = { .shared::cta }
.dsc = { .global }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.level::cache_hint = { .L2::cache_hint }
// global -> shared::cluster
cp.async.bulk.dst.srccompletion_mechanism {.multicast} {.level::cache_hint}
[dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}
.dst = { .shared::cluster }
.dsc = { .global }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.level::cache_hint = { .L2::cache_hint }
.multicast = { .multicast::cluster }
// shared::cta -> shared::cluster
cp.async.bulk.dst.srccompletion_mechanism [dstMem], [srcMem], size, [mbar]
.dst = { .shared::cluster }
.dsc = { .shared::cta }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
[/ shared::cta -> global
cp.async.bulk.dst.srccompletion_mechanism {.level::cache_hint} {.cp_mask}
[dstMem], [srcMem], size{, cache-policy} {, byteMask}
.dst = { .global }
.dsc = { .shared::cta }
.completion_mechanism = { .bulk_group }
.level::cache_hint = { .L2::cache_hint } 
```

# Description

cp.async.bulk is a non-blocking instruction which initiates an asynchronous bulk-copy operation from the location specified by source address operand srcMem to the location specified by destination address operand dstMem. 

The direction of bulk-copy is from the state space specified by the .src modifier to the state space specified by the .dst modifiers. 

The 32-bit operand size specifies the amount of memory to be copied, in terms of number of bytes. size must be a multiple of 16. If the value is not a multiple of 16, then the behavior is undefined. The memory range [dstMem, dstMem $^ +$ size - 1] must not overflow the destination memory space and the memory range [srcMem, srcMem $^ +$ size - 1] must not overflow the source memory space. Otherwise, the behavior is undefined. The addresses dstMem and srcMem must be aligned to 16 bytes. 

The optional qualifier .ignore_oob specifies that up to 15 bytes at the beginning or ending of [srcMem .. srcMem+size) may be out-of-bounds of a global memory allocation, and the value of the corresponding bytes in destination shared memory [dstMem .. dstMem+size) is indeterminate. The 32-bit operands ignoreBytesLeft and ignoreBytesRight are used to specify the bytes from beginning and ending of the copy-chunk specified by size that may go out of bounds. The only valid values for ignoreBytesLeft and ignoreBytesRight are [0..15], and any other value may result in undefined behavior. The srcMem and dstMem addresses must be aligned to 16 bytes, and the size operand must be a multiple of 16 even with .ignore_oob qualifier. The qualifier .ignore_oob is only available for the global to .shared::cta copy direction. 

When the destination of the copy is .shared::cta the destination address has to be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined. 

When the source of the copy is .shared::cta and the destination is .shared::cluster, the destination has to be in the shared memory of a different CTA within the cluster. 

The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. The completion mechanisms that are supported for different variants are summarized in the following table: 

<table><tr><td>.completion-mechanism</td><td>.dst</td><td>.src</td><td>Completion mechanism</td></tr><tr><td rowspan="3">.mbarrier::...</td><td>(shared::cta</td><td>.global</td><td rowspan="3">mbarrier based</td></tr><tr><td>.shared::cluster</td><td>.global</td></tr><tr><td>.shared::cluster</td><td>-shared::cta</td></tr><tr><td>.bulk_group</td><td>.global</td><td>-shared::cta</td><td>Bulk async-group based</td></tr></table>

The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk variant uses mbarrier based completion mechanism. The complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy. 

The modifier .bulk_group specifies that the cp.async.bulk variant uses bulk async-group based completion mechanism. 

The optional qualifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster. Operand ctaMask specifies the destination CTAs in the cluster such that each bit position in the 16-bit ctaMask operand corresponds to the %cluster_ctarank of the destination CTA. The source data is multicast to the same CTA-relative offset as dstMem in the shared memory of each destination CTA. The mbarrier signal is also multicast to the same CTA-relative offset as mbar in the shared memory of the destination CTA. 

When the optional argument cache-policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache-policy specifies the cache eviction policy that may be used during the memory access. 

cache-policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. The qualifier .level::cache_hint is only supported when at least one of the .src or .dst statespaces is .global state space. 

When the optional qualifier .cp_mask is specified, the argument byteMask is required. The i-th bit in the 16-bit wide byteMask operand specifies whether the i-th byte of each 16-byte wide chunk of source data is copied to the destination. If the bit is set, the byte is copied. 

The copy operation in cp.async.bulk is treated as a weak memory operation and the completetx operation on the mbarrier has .release semantics at the .cluster scope as described in the Memory Consistency Model. The copy operation is performed in the async proxy. 

# Notes

.multicast::cluster qualifier is optimized for target architecture sm_90a/sm_100f/sm_100a/ sm_103f/sm_103a/sm_110f/sm_110a and may have substantially reduced performance on other targets and hence .multicast::cluster is advised to be used with .target sm_90a/sm_100f/ sm_100a/sm_103f/sm_103a/sm_110f/sm_110a. 

# PTX ISA Notes

Introduced in PTX ISA version 8.0. 

Support for .shared::cta as destination state space is introduced in PTX ISA version 8.6. 

Support for .cp_mask qualifier introduced in PTX ISA version 8.6. 

Support for .ignore_oob qualifier introduced in PTX ISA version 9.2. 

# Target ISA Notes

Requires sm_90 or higher. 

.multicast::cluster qualifier advised to be used with .target sm_90a or sm_100f or sm_100a or sm_103f or sm_103a or sm_110f or sm_110a. 

Support for .cp_mask qualifier requires sm_100 or higher. 

# Examples

```txt
// .global -> .shared::cta (strictly non-remote):  
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];  
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [dstMem], [srcMem], size, [mbar], cache-policy;  
// .global -> .shared::cluster:  
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];  
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [mbar], ctaMask;  
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [dstMem], [srcMem], size, [mbar], cache-policy; 
```

(continues on next page) 

(continued from previous page) 

```asm
// .shared::cta -> .shared::cluster (strictly remote):  
cpasync.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dstMem], →[srcMem], size, [mbar];  
// .shared::cta -> .global:  
cpasync.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size;  
cpasync.bulk.global.shared::cta.bulk_group.L2::cache_hint} [dstMem], [srcMem], size, →cache-policy;  
// .shared::cta -> .global with .cp_mask:  
cpasync.bulk.global.shared::cta.bulk_group.L2::cache_hint.cp_mask [dstMem], [srcMem], →size, cache-policy, byteMask;  
// ignore_oob  
cpasync.bulk.shared::cta.global.mbarrier::complete_tx::bytesignore_oob [dstMem], →[srcMem], size, ignoreBytesLeft, ignoreBytesRight, [mbar]; 
```

# 9.7.9.25.4.2 Data Movement and Conversion Instructions: cp.reduce.async.bulk

cp.reduce.async.bulk 

Initiates an asynchronous reduction operation. 

# Syntax

```txt
cp.reduceasync.bulk.dst.srccompletion_mechanism.redOp.type [dstMem], [srcMem], size, [mbar] .dst = { .shared::cluster} .src = { .shared::cta} .completion_mechanism = { .mbarrier::complete_tx::bytes} .redOp= { .and, .or, .xor, .add, .inc, .dec, .min, .max} .type = { .b32, .u32, .s32, .b64, .u64} cp.reduceasync.bulk.dst.srccompletion_mechanism.{.level::cache_hint}.redOp.type [dstMem], [srcMem], size{, cache-policy} .dst = { .global} .src = { .shared::cta} .completion_mechanism = { .bulk_group} .level::cache_hint = { .L2::cache_hint} .redOp= { .and, .or, .xor, .add, .inc, .dec, .min, .max} .type = { .f16, .bf16, .b32, .u32, .s32, .b64, .u64, .s64, .f32, .f64 } 
```

(continues on next page) 

# 9.7. Instructions

(continued from previous page) 

```txt
cp.reduceasyncbulk.dst.src completion_mechanism{.level::cache_hint}.add.noftz.type [dstMem], [srcMem], size{, cache-policy} .dst = { .global } .src = { .shared::cta } .completion_mechanism = { .bulk_group } .type = { .f16, .bf16 } 
```

# Description

cp.reduce.async.bulk is a non-blocking instruction which initiates an asynchronous reduction operation on an array of memory locations specified by the destination address operand dstMem with the source array whose location is specified by the source address operand srcMem. The size of the source and the destination array must be the same and is specified by the operand size. 

Each data element in the destination array is reduced inline with the corresponding data element in the source array with the reduction operation specified by the modifier .redOp. The type of each data element in the source and the destination array is specified by the modifier .type. 

The source address operand srcMem is located in the state space specified by .src and the destination address operand dstMem is located in the state specified by the .dst. 

The 32-bit operand size specifies the amount of memory to be copied from the source location and used in the reduction operation, in terms of number of bytes. size must be a multiple of 16. If the value is not a multiple of 16, then the behavior is undefined. The memory range [dstMem, dstMem + size - 1] must not overflow the destination memory space and the memory range [srcMem, srcMem $^ +$ size - 1] must not overflow the source memory space. Otherwise, the behavior is undefined. The addresses dstMem and srcMem must be aligned to 16 bytes. 

The operations supported by .redOp are classified as follows: 

▶ The bit-size operations are .and, .or, and .xor. 

▶ The integer operations are .add, .inc, .dec, .min, and .max. The .inc and .dec operations return a result in the range [0..x] where x is the value at the source state space. 

▶ The floating point operation .add rounds to the nearest even. The current implementation of cp.reduce.async.bulk.add.f32 flushes subnormal inputs and results to sign-preserving zero. The cp.reduce.async.bulk.add.f16 and cp.reduce.async.bulk.add.bf16 operations require .noftz qualifier. It preserves input and result subnormals, and does not flush them to zero. 

The following table describes the valid combinations of .redOp and element type: 

<table><tr><td>.dst</td><td>.redOp</td><td>Element type</td></tr><tr><td rowspan="4">(shared::cluster</td><td>.add</td><td>.u32,.s32,.u64</td></tr><tr><td>.min,.max</td><td>.u32,.s32</td></tr><tr><td>.inc,.dec</td><td>.u32</td></tr><tr><td>.and,.or,.xor</td><td>.b32</td></tr><tr><td rowspan="4">.global</td><td>.add</td><td>.u32,.s32,.u64,.f32,.f64,.f16,.bf16</td></tr><tr><td>.min,.max</td><td>.u32,.s32,.u64,.s64,.f16,.bf16</td></tr><tr><td>.inc,.dec</td><td>.u32</td></tr><tr><td>.and,.or,.xor</td><td>.b32,.b64</td></tr></table>

The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. The completion mechanisms that are supported for different variants are summarized in the following table: 

<table><tr><td>.completion-mechanism</td><td>.dst</td><td>.src</td><td>Completion mechanism</td></tr><tr><td rowspan="2">.mbarrier::...</td><td>.shared::cluster</td><td>.global</td><td rowspan="2">mbarrier based</td></tr><tr><td>.shared::cluster</td><td>.shared::cta</td></tr><tr><td>.bulk_group</td><td>.global</td><td>.shared::cta</td><td>Bulk async-group based</td></tr></table>

The modifier .mbarrier::complete_tx::bytes specifies that the cp.reduce.async.bulk variant uses mbarrier based completion mechanism. The complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy. 

The modifier .bulk_group specifies that the cp.reduce.async.bulk variant uses bulk async-group based completion mechanism. 

When the optional argument cache-policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache-policy specifies the cache eviction policy that may be used during the memory access. 

cache-policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. The qualifier .level::cache_hint is only supported when at least one of the .src or .dst statespaces is .global state space. 

Each reduction operation performed by the cp.reduce.async.bulk has individually .relaxed.gpu memory ordering semantics. The load operations in cp.reduce.async.bulk are treated as weak memory operation and the complete-tx operation on the mbarrier has .release semantics at the . cluster scope as described in the Memory Consistency Model. The memory operations are performed in the async proxy. 

# PTX ISA Notes

Introduced in PTX ISA version 8.0. 

# Target ISA Notes

Requires sm_90 or higher. 

# Examples

cp.reduceasync.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64 [dstMem], [srcMem], $\rightarrow$ size, [mbar];   
cp.reduceasync.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.s32 [dstMem], [srcMem], $\rightarrow$ size, [mbar];   
cp.reduceasync.bulk.global.shared::cta.bulk_group.min.f16 [dstMem], [srcMem], size;   
cp.reduceasync.bulk.global.shared::cta.bulk_group.L2::cache_hint.xor.s32 [dstMem], $\rightarrow$ [srcMem], size, policy;   
cp.reduceasync.bulk.global.shared::cta.bulk_group.add.noftz.f16 [dstMem], [srcMem], $\rightarrow$ size; 

# 9.7.9.25.4.3 Data Movement and Conversion Instructions: cp.async.bulk.prefetch

cp.async.bulk.prefetch 

Provides a hint to the system to initiate the asynchronous prefetch of data to the cache. 

# Syntax

```txt
cpasync.bulk.prefetch.L2.src{.level::cache_hint} [srcMem], size {, cache-policy};  
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
```
`` 
```

# Description

cp.async.bulk.prefetch is a non-blocking instruction which may initiate an asynchronous prefetch of data from the location specified by source address operand srcMem, in .src statespace, to the L2 cache. 

The 32-bit operand size specifies the amount of memory to be prefetched in terms of number of bytes. size must be a multiple of 16. If the value is not a multiple of 16, then the behavior is undefined. The address srcMem must be aligned to 16 bytes. 

When the optional argument cache-policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache-policy specifies the cache eviction policy that may be used during the memory access. 

cache-policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. 

cp.async.bulk.prefetch is treated as a weak memory operation in the Memory Consistency Model. 

# PTX ISA Notes

Introduced in PTX ISA version 8.0. 

# Target ISA Notes

Requires sm_90 or higher. 

# Examples

```txt
cpasync.bulk sufetch.L2.global [srcMem], size;   
cpasync.bulk sufetch.L2.global.L2::cache_hint [srcMem], size, policy; 
```

# 9.7.9.25.4.4 Data Movement and Conversion Instructions: multimem.cp.async.bulk

multimem.cp.async.bulk 

Initiates an asynchronous copy operation to a multimem address range. 

# Syntax

```txt
multimem.cpasync.bulk.dst.src completion_mechanism{.cp_mask} [dstMem], [srcMem], size{, byteMask}; .dst = { .global } .src = { .shared::cta } .completion_mechanism = { .bulk_group } 
```

# Description

Instruction multimem.cp.async.bulk initiates an asynchronous bulk-copy operation from source address range [srcMem, srcMem $^ +$ size) to memory locations residing on each GPU’s memory referred to by the destination multimem address range [dstMem, dstMem $^ +$ size). The direction of bulk-copy is from the state space specified by the .src modifier to the state space specified by the .dst modifiers. 

The 32-bit operand size specifies the amount of memory to be copied, in terms of number of bytes. Operand size must be a multiple of 16. The memory range [dstMem, dstMem $^ +$ size) must not 

overflow the destination multimem memory space. The memory range [srcMem, srcMem $^ +$ size) must not overflow the source memory space. The addresses dstMem and srcMem must be aligned to 16 bytes. If any of these pre-conditions is not met, the behavior is undefined. 

The modifier .completion_mechanism specifies the completion mechanism that is supported by the instruction. The modifier .bulk_group specifies that the multimem.cp.async.bulk instruction uses bulk async-group based completion mechanism. 

When the optional modifier .cp_mask is specified, the argument byteMask is required. The i-th bit in the 16-bit wide byteMask operand specifies whether the i-th byte of each 16-byte wide chunk of source data is copied to the destination. If the bit is set, the byte is copied. 

The reads and writes of the copy operation in multimem.cp.async.bulk are weak memory operations as described in the Memory Consistency Model. 

# PTX ISA Notes

Introduced in PTX ISA version 9.1. 

# Target ISA Notes

Requires sm_90 or higher. 

Support for .cp_mask qualifier requires sm_100 or higher. 

# Examples

multimem.cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size; 

multimem.cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], 512; 

multimem.cp.async.bulk.global.shared::cta.bulk_group.cp_mask [dstMem], [srcMem], size, $\cdot$ byteMask; 

# 9.7.9.25.4.5 Data Movement and Conversion Instructions: multimem.cp.reduce.async.bulk

multimem.cp.reduce.async.bulk 

Initiates an asynchronous reduction operation to a multimem address range. 

# Syntax

multimem.cp.reduce.async.bulk.dst.src.completion_mechanism.redOp.type [dstMem], $\cdot$ [srcMem], size; 

```lua
.dst = { .global }  
.src = { .shared::cta }  
Completion_mechanism = { .bulk_group }  
.redOp = { .and, .or, .xor, .add, .inc, .dec, 
```

(continues on next page) 

(continued from previous page) 

.min，.max} .type $= \{$ .f16，.bf16, .b32，.u32，.s32, .b64，.u64，.s64, .f32，.f64} 

```txt
multimem.mp.reduceasync.bulk.dst.srccompletion_mechanism.add.noftz.type [dstMem], →[srcMem], size; 
```

```txt
.dst = { .global }  
.src = { .shared::cta }  
Completion_mechanism = { .bulk_group }  
.type = { .f16, .bf16 } 
```

# Description

Instruction multimem.cp.reduce.async.bulk initiates an element-wise asynchronous reduction operation with elements from source memory address range [srcMem, srcMem $^ +$ size) to memory locations residing on each GPU’s memory referred to by the multimem destination address range [dstMem, dstMem $^ +$ size). 

Each data element in the destination array is reduced inline with the corresponding data element in the source array with the reduction operation specified by the modifier .redOp. The type of each data element in the source and the destination array is specified by the modifier .type. 

The source address operand srcMem is in the state space specified by .src and the destination address operand dstMem is in the state specified by the .dst. 

The 32-bit operand size specifies the amount of memory to be copied from the source location and used in the reduction operation, in terms of number of bytes. Operand size must be a multiple of 16. The memory range [dstMem, dstMem $^ +$ size) must not overflow the destination multimem memory space. The memory range [srcMem, srcMem $^ +$ size) must not overflow the source memory space. The addresses dstMem and srcMem must be aligned to 16 bytes. If any of these preconditions is not met, the behavior is undefined. 

The operations supported by .redOp are classified as follows: 

The bit-size operations are .and, .or, and .xor. 

The integer operations are .add, .inc, .dec, .min, and .max. The .inc and .dec operations return a result in the range [0..x] where x is the value at the source state space. 

The floating point operation .add rounds to the nearest even, preserve input and result subnormals, and does not flush them to zero, except for the current implementation of multimem.cp. reduce.async.bulk.add.f32 which flushes subnormal inputs and results to sign-preserving zero. The multimem.cp.reduce.async.bulk.add.f16 and multimem.cp.reduce.async.bulk.add. bf16 operations require .noftz qualifier. It preserves input and result subnormals, and does not flush them to zero. 

The following table describes the valid combinations of .redOp and element type: 

<table><tr><td>.redOp</td><td>element type</td></tr><tr><td>.add</td><td>.u32, .s32, .u64, .f32, .f64, .f16, .bf16</td></tr><tr><td>.min, .max</td><td>.u32, .s32, .u64, .s64, .f16, .bf16</td></tr><tr><td>.inc, .dec</td><td>.u32</td></tr><tr><td>.and, .or, .xor</td><td>.b32, .b64</td></tr></table>

The modifier .completion_mechanism specifies the completion mechanism that is supported by the instruction. The modifier .bulk_group specifies that the multimem.cp.reduce.async.bulk uses bulk async-group based completion mechanism. 

Each reduction operation performed by the multimem.cp.reduce.async.bulk has individually . relaxed.sys memory ordering semantics. The load operations in multimem.cp.reduce.async. bulk are treated as weak memory operations as described in the Memory Consistency Model. 

# PTX ISA Notes

Introduced in PTX ISA version 9.1. 

# Target ISA Notes

Requires sm_90 or higher. 

# Examples

multimem.cp.reduceasync.bulk.global.shared::cta.bulk_group.add.u32 [dstMem], $\rightarrow$ [srcMem], size;   
multimem.cp.reduceasync.bulk.global.shared::cta.bulk_group.xor.b64 [dstMem], $\rightarrow$ [srcMem], size;   
multimem.cp.reduceasync.bulk.global.shared::cta.bulk_group.inc.u32 [dstMem], $\rightarrow$ [srcMem], size;   
multimem.cp.reduceasync.bulk.global.shared::cta.bulk_group.dec.u32 [dstMem], $\rightarrow$ [srcMem], size;   
multimem.cp.reduceasync.bulk.global.shared::cta.bulk_group.max.s32 [dstMem], $\rightarrow$ [srcMem], size;   
multimem.cp.reduceasync.bulk.global.shared::cta.bulk_group.add.noftz.f16 [dstMem], $\rightarrow$ [srcMem], size;   
multimem.cp.reduceasync.bulk.global.shared::cta.bulk_group.min.cf16 [dstMem], $\rightarrow$ [srcMem], size;   
multimem.cp.reduceasync.bulk.global.shared::cta.bulk_group.add.noftz.cf16 [dstMem], $\rightarrow$ [srcMem], size; 

# 9.7.9.25.5 Data Movement and Conversion Instructions: Tensor copy

# 9.7.9.25.5.1 Restriction on Tensor Copy instructions

Following are the restrictions on the types .b4x16, .b4x16_p64, .b6x16_p32 and .b6p2x16: 

1. cp.reduce.async.bulk doesn’t support the types .b4x16, .b4x16_p64, .b6x16_p32 and . b6p2x16. 

2. cp.async.bulk.tensor with the direction .global.shared::cta doesn’t support the type .b4x16_p64. 

3. cp.async.bulk.tensor with the direction .shared::cluster.global doesn’t support the sub-byte types on sm_120a. 

4. OOB-NaN fill mode doesn’t support the types .b4x16, .b4x16_p64, .b6x16_p32 and .b6p2x16. 

5. Box-Size[0] must be exactly: 

1. 96B for b6x16_p32 and .b6p2x16. 

2. 64B for b4x16_p64. 

6. Tensor-Size[0] must be a multiple of: 

1. 96B for b6x16_p32 and .b6p2x16. 

2. 64B for b4x16_p64. 

7. For .b4x16_p64, .b6x16_p32 and . ${ \mathsf { b 6 } } { \mathsf { p } } 2 { \mathsf { x } } 1 6$ , the first coordinate in the tensorCoords argument vector must be a multiple of 128. 

8. For .b4x16_p64, .b6x16_p32 and . ${ \mathsf { b } } 6 { \mathsf { p } } 2 { \mathsf { x } } 1 6$ , the global memory address must be 32B aligned. Additionally, tensor stride in every dimension must be 32B aligned. 

9. .b4x16_p64, .b6x16_p32 and . ${ \mathsf { b } } 6 { \mathsf { p } } 2 { \mathsf { x } } 1 6$ supports the following swizzling modes: 

1. None. 

2. 128B (With all potential swizzle atomicity values except: 32B with 8B flip) 

Following are the restrictions on the 96B swizzle mode: 

1. The .swizzle_atomicity must be 16B. 

2. The .interleave_layout must not be set. 

3. Box-Size[0] must be less than or equal to 96B. 

4. The type must not be among following: .b4x16_p64, .b6x16_p32 and .b6p2x16. 

5. The .load_mode must not be set to .im2col::w::128. 

Following are the restrictions on the 128-byte swizzle mode having 32-byte atomicity with 8-byte flip as sub-mode: 

1. The copy direction must be .shared::cluster.global or otherwise .shared::cta.global. 

2. The .load_mode must not be set to .im2col::w, .im2col::w::128. 

Following are the restrictions on the .global.shared::cta direction: 

1. Starting co-ordinates for Bounding Box (tensorCoords) must be non-negative. 

2. The bounding box along the D, W and H dimensions must stay within the tensor boundaries. This implies: 

1. Bounding-Box Lower-Corner must be non-negative. 

2. Bounding-Box Upper-Corner must be non-positive. 

Following are the restrictions for sm_120a: 

1. cp.async.bulk.tensor with the direction .shared::cluster.global doesn’t support: 

1. the sub-byte types 

2. the qualifier .swizzle_atomicity 

Following are the restrictions for sm_103a while using type .b6p2x16 on cp.async.bulk.tensor with the direction .global.shared::cta: 

1. Box-Size[0] must be exactly either of 48B or 96B. 

2. The global memory address must be 16B aligned. 

3. Tensor Stride in every dimension must be 16B aligned. 

4. The first coordinate in the tensorCoords argument vector must be a multiple of 64. 

5. Tensor-Size[0] must be a multiple of 48B. 

6. The following swizzle modes are supported: 

1. None. 

2. 128B (With all potential swizzle atomicity values except: 32B with 8B flip) 

3. 64B swizzle with 16B swizzle atomicity 

# 9.7.9.25.5.2 Data Movement and Conversion Instructions: cp.async.bulk.tensor

# cp.async.bulk.tensor

Initiates an asynchronous copy operation on the tensor data from one state space to another. 

# Syntax

```rust
// global -> shared::cta
cpasync.bulk>tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}.
→level::cache_hint}
[dstMem], [tensorMap, tensorCoords], [mbar],
→im2colInfo} {, cache-policy}
.dst = { .shared::cta }
.src = { .global }
.dim = { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group = { .cta_group::1, .cta_group::2 }
.load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128
→}
.level::cache_hint = { .L2::cache_hint }
// global -> shared::cluster
cpasync.bulk>tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_→group} {.level::cache_hint} 
```

(continues on next page) 

(continued from previous page) 

```ini
[dstMem], [tensorMap, tensorCoords], [mbar],
→im2colInfo}
{, ctaMask} {, cache-policy}
.dst = { .shared::cluster }
.src = { .global }
.dim = { .1d, .2d, .3d, .4d, .5d }
completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group = { .cta_group::1, .cta_group::2 }
.load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128
→}
.level::cache_hint = { .L2::cache_hint }
.multicast = { .multicast::cluster }
// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src {.load_mode}.completion_mechanism {.level::cache_hint}
→policy}
.dst = { .global }
.src = { .shared::cta }
.dim = { .1d, .2d, .3d, .4d, .5d }
顺利完成_mechanism = { .bulk_group }
.load_mode = { .tile, .tile::scatter4, .im2col_no_offs }
.level::cache_hint = { .L2::cache_hint }
```

# Description

cp.async.bulk.tensor is a non-blocking instruction which initiates an asynchronous copy operation of tensor data from the location in .src state space to the location in the .dst state space. 

The operand dstMem specifies the location in the .dst state space into which the tensor data has to be copied and srcMem specifies the location in the .src state space from which the tensor data has to be copied. 

When .dst is specified as .shared::cta, the address dstMem must be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined. 

When .dst is specified as .shared::cluster, the address dstMem can be in the shared memory of any of the CTAs within the current cluster. 

The operand tensorMap is the generic address of the opaque tensor-map object which resides in . param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation, as described in Tensor-map. The tensorMap is accessed in tensormap proxy. Refer to the CUDA programming guide for creating the tensor-map objects on the host side. 

The dimension of the tensor data is specified by the .dim modifier. 

The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory from or to which the copy operation has to be performed. The individual tensor coordinates in tensorCoords are of type .s32. The format of vector argument tensorCoords is dependent on .load_mode specified and is as follows: 

<table><tr><td>.load_mode</td><td>tensorCoords</td><td>Semantics</td></tr><tr><td>.tile::scatter4</td><td rowspan="2">{colidx, rowidx0, rowidx1, rowidx2, rowidx3}</td><td rowspan="2">Fixed length vector of size 5. The five elements together specify the start coordinates of the four rows.</td></tr><tr><td>.tile::gather4</td></tr><tr><td>Rest all</td><td>{d0, ..., dn} for n = .dim</td><td>Vector of n elements where n = .dim. The elements indicate the offset in each of the dimension.</td></tr></table>

The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. The completion mechanisms that are supported for different variants are summarized in the following table: 

<table><tr><td rowspan="2">.completion-mechanism</td><td rowspan="2">.dst</td><td rowspan="2">.src</td><td colspan="2">Completion mechanism</td></tr><tr><td>Needed for completion of entireAsync operation</td><td>optionally can be used for the completion of reading of the tensorsmap object</td></tr><tr><td rowspan="2">.mbarrier::...</td><td>.shared::cta</td><td>.global</td><td rowspan="2">mbarrier based</td><td rowspan="3">Bulk async-group based</td></tr><tr><td>.shared::cluster</td><td>.global</td></tr><tr><td>.bulk_group</td><td>.global</td><td>.shared::cta</td><td>Bulk async-group based</td></tr></table>

The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk.tensor variant uses mbarrier based completion mechanism. Upon the completion of the asynchronous copy operation, the complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy. 

The modifier .cta_group can only be specified with the mbarrier based completion mechanism. The modifier .cta_group is used to signal either the odd numbered CTA or the even numbered CTA among the CTA-Pair. When .cta_group::1 is specified, the mbarrier object mbar that is specified must be in the shared memory of the same CTA as the shared memory destination dstMem. When .cta_group::2 is specified, the mbarrier object mbar can be in shared memory of either the same CTA as the shared memory destination dstMem or in its peer-CTA. If .cta_group is not specified, then it defaults to .cta_group::1. 

The modifier .bulk_group specifies that the cp.async.bulk.tensor variant uses bulk async-group based completion mechanism. 

The qualifier .load_mode specifies how the data in the source location is copied into the destination location. If .load_mode is not specified, it defaults to .tile. 

In .tile mode, the multi-dimensional layout of the source tensor is preserved at the destination. 

In .tile::gather4 mode, four rows in 2-dimnesional source tensor are combined to form a single 2-dimensional destination tensor. In .tile::scatter4 mode, single 2-dimensional source tensor is divided into four rows in 2-dimensional destination tensor. Details of .tile::scatter4/. tile::gather4 modes are described in .tile::scatter4 and .tile::gather4 modes. 

In .im2col and .im2col::* modes, some dimensions of the source tensors are unrolled in a single dimensional column at the destination. Details of the im2col and .im2col::* modes are described in im2col mode and im2col::w and im2col::w::128 modes respectively. In .im2col and .im2col::* modes, the tensor has to be at least 3-dimensional. The vector operand im2colInfo can be specified only when .load_mode is .im2col or .im2col::w or .im2col::w::128. The format of the vector argument im2colInfo is dependent on the exact im2col mode and is as follows: 

<table><tr><td>Exact im2col mode</td><td>im2collinfo argument</td><td>Semantics</td></tr><tr><td>.im2col</td><td>{i2cOffW, i2cOffH, i2cOffD} for .dim = .5d</td><td>A vector of im2col offsets whose vector size is two less than number of dimensions .dim.</td></tr><tr><td>.im2col::w</td><td rowspan="2">{wHalo, wOffset}</td><td rowspan="2">A vector of 2 arguments containing wHalo and wOffset arguments.</td></tr><tr><td>.im2col::w::128</td></tr><tr><td>.im2col_no_offs</td><td>im2colInfo is not applicable.</td><td>im2colInfo is not applicable.</td></tr></table>

Argument wHalo is a 16bit unsigned integer whose valid set of values differs on the load-mode and is as follows: - Im2col::w mode : valid range is [0, 512). - Im2col::w::128 mode : valid range is [0, 32). 

Argument wOffset is a 16bit unsigned integer whose valid range of values is [0, 32). 

The optional qualifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster. Operand ctaMask specifies the destination CTAs in the cluster such that each bit position in the 16-bit ctaMask operand corresponds to the %cluster_ctarank of the destination CTA. The source data is multicast to the same offset as dstMem in the shared memory of each destination CTA. When .cta_group is specified as: 

▶ .cta_group::1 : The mbarrier signal is also multicasted to the same offset as mbar in the shared memory of the destination CTA. 

▶ .cta_group::2 : The mbarrier signal is multicasted either to all the odd numbered CTAs or the even numbered CTAs within the corresponding CTA-Pair. For each destination CTA specified in the ctaMask, the mbarrier signal is sent either to the destination CTA or its peer-CTA based on CTAs %cluster_ctarank parity of shared memory where the mbarrier object mbar resides. 

When the optional argument cache-policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache-policy specifies the cache eviction policy that may be used during the memory access. 

cache-policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. 

The copy operation in cp.async.bulk.tensor is treated as a weak memory operation and the complete-tx operation on the mbarrier has .release semantics at the .cluster scope as described in the Memory Consistency Model. 

# Notes

.multicast::cluster qualifier is optimized for target architecture sm_90a/sm_100f/sm_100a/ sm_103f/sm_103a/sm_110f/sm_110a and may have substantially reduced performance on other targets and hence .multicast::cluster is advised to be used with .target sm_90a/sm_100f/ sm_100a/sm_103f/sm_103a/sm_110f/sm_110a. 

# PTX ISA Notes

Introduced in PTX ISA version 8.0. 

Support for .shared::cta as destination state space is introduced in PTX ISA version 8.6. 

Support for qualifiers .tile::gather4 and .tile::scatter4 introduced in PTX ISA version 8.6. 

Support for qualifiers .im2col::w and .im2col::w::128 introduced in PTX ISA version 8.6. 

Support for qualifier .cta_group introduced in PTX ISA version 8.6. 

# Target ISA Notes

Requires sm_90 or higher. 

.multicast::cluster qualifier advised to be used with .target sm_90a or sm_100f or sm_100a or sm_103f or sm_103a or sm_110f or sm_110a. 

Qualifiers .tile::gather4 and .im2col::w require: 

▶ sm_100a when destination state space is .shared::cluster and is supported on sm_100f from PTX ISA version 8.8. 

▶ sm_100 or higher when destination state space is .shared::cta. 

Qualifier .tile::scatter4 is supported on following architectures: 

▶ sm_100a 

▶ sm_101a (Renamed to sm_110a from PTX ISA version 9.0) 

▶ And is supported on following family-specific architectures from PTX ISA version 8.8: 

▶ sm_100f or higher in the same family 

▶ sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0) 

▶ sm_110f or higher in the same family 

Qualifier .im2col::w::128 is supported on following architectures: 

▶ sm_100a 

▶ sm_101a (Renamed to sm_110a from PTX ISA version 9.0) 

▶ And is supported on following family-specific architectures from PTX ISA version 8.8: 

▶ sm_100f or higher in the same family 

▶ sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0) 

▶ sm_110f or higher in the same family 

Qualifier .cta_group is supported on following architectures: 

▶ sm_100a 

▶ sm_101a (Renamed to sm_110a from PTX ISA version 9.0) 

▶ And is supported on following family-specific architectures from PTX ISA version 8.8: 

▶ sm_100f or higher in the same family 

▶ sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0) 

▶ sm_110f or higher in the same family 

# Examples

.reg .b16 ctaMask;   
.reg .u16 i2cOffW, i2cOffH, i2cOffD;   
.reg .b64 l2CachePolicy;   
cpasync.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes tile [sMem0], $\rightarrow$ [tensorMap0, {tc0}], [mbar0];   
@p cpasync.bulk.tensor.5d.shared::cta.global.im2col.mbarrier::complete_tx::bytes [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], $\rightarrow$ {i2cOffW, i2cOffH, i2cOffD};   
cpasync.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytestile $\rightarrow$ [sMem0], [tensorMap0, {tc0}], [mbar0];   
@p cpasync.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes. $\rightarrow$ multicast::cluster [sMem1], [tensorMap1, {tc0, tc1}], [mbar2], ctaMask;   
@p cpasync.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], $\rightarrow$ {i2cOffW, i2cOffH, i2cOffD};   
@p cpasync.bulk.tensor.3d.im2col.shared::cluster.global.mbarrier::complete_tx::bytes. $\rightarrow$ L2::cache_hint [sMem3], [tensorMap3, {tc0, tc1, tc2}], [mbar3], {i2cOffW}, policy;   
@p cpasync.bulk.tensor.1d.global.shared::cta*bulk_group [tensorMap3, {tc0}], $\rightarrow$ [sMem3];   
cpasync.bulk.tensor.2dtile::gather4.shared::cluster.global.mbarrier::complete_ $\rightarrow$ tx::bytes [sMem5], [tensorMap6, {x0, y0, y1, y2, y3}], [mbar5];   
cpasync.bulk.tensor.3d.im2col::w.shared::cluster.global.mbarrier::complete_tx::bytes [sMem4], [tensorMap5, {t0, t1, t2}], [mbar4], {im2colwHalo, $\rightarrow$ im2coloff};   
cpasync.bulk.tensor.1d.shared::cluster.globaltile.cta_group::2 [sMem6], [tensorMap7, {tc0}], [peerMbar]; 

# 9.7.9.25.5.3 Data Movement and Conversion Instructions: cp.reduce.async.bulk.tensor

# cp.reduce.async.bulk.tensor

Initiates an asynchronous reduction operation on the tensor data. 

# Syntax

```txt
// shared::cta -> global:  
cp.reduce.async.bulk.tensor.dim.dst.src.redOp {.load_mode}.completion_mechanism{.  
→level::cache_hint} [tensorMap, tensorCoords], [srcMem] {,cache-  
→policy}  
.dst = { .global }  
.src = { .shared::cta }  
.dim = { .1d, .2d, .3d, .4d, .5d }  
.completion_mechanism = { .bulk_group }  
.load_mode = { .tile, .im2col_no_offs }  
.redOp = { .add, .min, .max, .inc, .dec, .and, .or, .xor} 
```

# Description

cp.reduce.async.bulk.tensor is a non-blocking instruction which initiates an asynchronous reduction operation of tensor data in the .dst state space with tensor data in the .src state space. 

The operand srcMem specifies the location of the tensor data in the .src state space using which the reduction operation has to be performed. 

The operand tensorMap is the generic address of the opaque tensor-map object which resides in . param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation, as described in Tensor-map. The tensorMap is accessed in tensormap proxy. Refer to the CUDA programming guide for creating the tensor-map objects on the host side. 

Each element of the tensor data in the .dst state space is reduced inline with the corresponding element from the tensor data in the .src state space. The modifier .redOp specifies the reduction operation used for the inline reduction. The type of each tensor data element in the source and the destination tensor is specified in Tensor-map. 

The dimension of the tensor is specified by the .dim modifier. 

The vector operand tensorCoords specifies the starting coordinates of the tensor data in the global memory on which the reduce operation is to be performed. The number of tensor coordinates in the vector argument tensorCoords should be equal to the dimension specified by the modifier .dim. The individual tensor coordinates are of the type .s32. 

The following table describes the valid combinations of .redOp and element type: 

<table><tr><td>.redOp</td><td>Element type</td></tr><tr><td>.add</td><td>.u32, .s32, .u64, .f32, .f16, .bf16</td></tr><tr><td>.min, .max</td><td>.u32, .s32, .u64, .s64, .f16, .bf16</td></tr><tr><td>.inc, .dec</td><td>.u32</td></tr><tr><td>.and, .or, .xor</td><td>.b32, .b64</td></tr></table>

The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. Value .bulk_group of the modifier .completion_mechanism specifies that cp.reduce.async.bulk.tensor instruction uses bulk async-group based completion mechanism. 

The qualifier .load_mode specifies how the data in the source location is copied into the destination location. If .load_mode is not specified, it defaults to .tile. In .tile mode, the multi-dimensional layout of the source tensor is preserved at the destination. In .im2col_no_offs mode, some dimensions of the source tensors are unrolled in a single dimensional column at the destination. Details of the im2col mode are described in im2col mode. In .im2col mode, the tensor has to be at least 3-dimensional. 

When the optional argument cache-policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache-policy specifies the cache eviction policy that may be used during the memory access. 

cache-policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. The qualifier .level::cache_hint is only supported when at least one of the .src or .dst statespaces is .global state space. 

Each reduction operation performed by cp.reduce.async.bulk.tensor has individually . relaxed.gpu memory ordering semantics. The load operations in cp.reduce.async.bulk.tensor are treated as weak memory operations and the complete-tx operation on the mbarrier has .release semantics at the .cluster scope as described in the Memory Consistency Model. 

# PTX ISA Notes

Introduced in PTX ISA version 8.0. 

# Target ISA Notes

Requires sm_90 or higher. 

# Examples

cp.reduce_async.bulk.tensor.1d.global.shared::cta.addtile.bulk_group [tensorMap0,{tc0}],[sMem0];   
cp.reduce_async.bulk.tensor.2d.global.shared::cta.and.bulk_group.L2::cache_hint [tensorMap1,{tc0,tc1}],[sMem1], $\rightarrow$ policy;   
cp.reduce_async.bulk.tensor.3d.global.shared::cta.xor.im2col;bulk_group [tensorMap2,{tc0,tc1,tc2}],[sMem2]; 

# 9.7. Instructions

# 9.7.9.25.5.4 Data Movement and Conversion Instructions: cp.async.bulk.prefetch.tensor

# cp.async.bulk.prefetch.tensor

Provides a hint to the system to initiate the asynchronous prefetch of tensor data to the cache. 

# Syntax

// global -> L2:  
cp.async.bulk sufetch.tensor.dim.L2.src{.load_mode} {.level::cache_hint} [tensorMap, $\rightarrow$ tensorCoords]  
{，im2colInfo} {，cache- $\rightarrow$ policy}  
. src = { .global }  
. dim = { .1d, .2d, .3d, .4d, .5d }  
. load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }  
. level::cache_hint = { .L2::cache_hint } 

# Description

cp.async.bulk.prefetch.tensor is a non-blocking instruction which may initiate an asynchronous prefetch of tensor data from the location in .src statespace to the L2 cache. 

The operand tensorMap is the generic address of the opaque tensor-map object which resides in . param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation, as described in Tensor-map. The tensorMap is accessed in tensormap proxy. Refer to the CUDA programming guide for creating the tensor-map objects on the host side. 

The dimension of the tensor data is specified by the .dim modifier. 

The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory from which the copy operation has to be performed. The individual tensor coordinates in tensorCoords are of type .s32. The format of vector argument tensorCoords is dependent on .load_mode specified and is as follows: 

<table><tr><td>.load_mode</td><td>tensorCoords</td><td>Semantics</td></tr><tr><td>.tile::gather4</td><td>{colidx, rowidx0, rowidx1, rowidx2, rowidx3}</td><td>Fixed length vector of size 5. The five elements together specify the start coordinates of the four rows.</td></tr><tr><td>Rest all</td><td>{d0,.,dn} for n = .dim</td><td>Vector of n elements where n = .dim. The elements indicate the offset in each of the dimension.</td></tr></table>

The qualifier .load_mode specifies how the data in the source location is copied into the destination location. If .load_mode is not specified, it defaults to .tile. 

In .tile mode, the multi-dimensional layout of the source tensor is preserved at the destination. In .tile::gather4 mode, four rows in the 2-dimnesional source tensor are fetched to L2 cache. Details of .tile::gather4 modes are described in .tile::scatter4 and .tile::gather4 modes. 

In .im2col and .im2col::* modes, some dimensions of the source tensors are unrolled in a single dimensional column at the destination. Details of the im2col and .im2col::* modes are described in im2col mode and im2col::w and im2col::w::128 modes respectively. In .im2col and .im2col::* modes, the tensor has to be at least 3-dimensional. The vector operand im2colInfo can be specified only when .load_mode is .im2col or .im2col::w or .im2col::w::128. The format of the vector argument im2colInfo is dependent on the exact im2col mode and is as follows: 

<table><tr><td>Exact im2col mode</td><td>im2collinfo argument</td><td>Semantics</td></tr><tr><td>.im2col</td><td>{i2cOffW, i2cOffH, i2cOffD} for .dim = .5d</td><td>A vector of im2col offsets whose vector size is two less than number of dimensions .dim.</td></tr><tr><td>.im2col::w</td><td rowspan="2">{wHalo, wOffset}</td><td rowspan="2">A vector of 2 arguments containing wHalo and wOffset arguments.</td></tr><tr><td>.im2col::w::128</td></tr></table>

When the optional argument cache-policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache-policy specifies the cache eviction policy that may be used during the memory access. 

cache-policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program. 

cp.async.bulk.prefetch.tensor is treated as a weak memory operation in the Memory Consistency Model. 

# PTX ISA Notes

Introduced in PTX ISA version 8.0. 

Support for qualifier .tile::gather4 introduced in PTX ISA version 8.6. 

Support for qualifiers .im2col::w and .im2col::w::128 introduced in PTX ISA version 8.6. 

# Target ISA Notes

Requires sm_90 or higher. 

Qualifier .tile::gather4 is supported on following architectures: 

▶ sm_100a 

▶ sm_101a (Renamed to sm_110a from PTX ISA version 9.0) 

▶ And is supported on following family-specific architectures from PTX ISA version 8.8: 

▶ sm_100f or higher in the same family 

▶ sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0) 

▶ sm_110f or higher in the same family 

Qualifiers .im2col::w and .im2col::w::128 are supported on following architectures: 

▶ sm_100a 

▶ sm_101a (Renamed to sm_110a from PTX ISA version 9.0) 

▶ And are supported on following family-specific architectures from PTX ISA version 8.8: 

▶ sm_100f or higher in the same family 

▶ sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0) 

▶ sm_110f or higher in the same family 

# Examples

.reg .b16 ctaMask, im2colwHalo, im2col10ff;   
.reg .u16 i2cOffW, i2cOffH, i2cOffD;   
.reg .b64 l2CachePolicy;   
cp.async.bulk PREFetch.tensor.1d.L2.globaltile [tensorMap0, {tc0}];   
@p cp.async.bulk PREFetch.tensor.2d.L2.global [tensorMap1, {tc0, tc1}];   
@p cp.async.bulk PREFetch.tensor.5d.L2.global.im2col [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], {i2cOffW, i2cOffH, $\rightarrow$ i2cOffD};   
@p cp.async.bulk PREFetch.tensor.3d.L2.global.im2col.L2::cache_hint [tensorMap3, {tc0, tc1, tc2}], {i2cOffW}, policy;   
cp.async.bulk PREFetch.tensor.2d.L2.globaltile::gather4 [tensorMap5, {colidx, row_ $\rightarrow$ idx0, row_idx1, row_idx2, row_idx3}];   
cp.async.bulk PREFetch.tensor.4d.L2.global.im2col::w::128 [tensorMap4, {t0, t1, t2, t3}], {im2colwHalo, im2col10ff}; 

# 9.7.9.25.6 Data Movement and Conversion Instructions: Bulk and Tensor copy completion instructions

# 9.7.9.25.6.1 Data Movement and Conversion Instructions: cp.async.bulk.commit_group

cp.async.bulk.commit_group 

Commits all prior initiated but uncommitted cp.async.bulk instructions into a cp.async.bulk-group. 

# Syntax

cp.async.bulk.commit_group; 

# Description

cp.async.bulk.commit_group instruction creates a new per-thread bulk async-group and batches all prior cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions satisfying the following conditions into the new bulk async-group: 

▶ The prior cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions use bulk_group based completion mechanism, and 

▶ They are initiated by the executing thread but not committed to any bulk async-group. 

If there are no uncommitted cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions then cp.async.bulk.commit_group results in an empty bulk async-group. 

An executing thread can wait for the completion of all cp{.reduce}.async.bulk{.prefetch}{. tensor} operations in a bulk async-group using cp.async.bulk.wait_group. 

There is no memory ordering guarantee provided between any two cp{.reduce}.async.bulk{. prefetch}{.tensor} operations within the same bulk async-group. 

# PTX ISA Notes

Introduced in PTX ISA version 8.0. 

# Target ISA Notes

Requires sm_90 or higher. 

# Examples

cp.async.bulk.commit_group; 

9.7.9.25.6.2 Data Movement and Conversion Instructions: cp.async.bulk.wait_group 

cp.async.bulk.wait_group 

Wait for completion of bulk async-groups. 

# Syntax

cp.async.bulk.wait_group{.read} N; 

# Description

cp.async.bulk.wait_group instruction will cause the executing thread to wait until only N or fewer of the most recent bulk async-groups are pending and all the prior bulk async-groups committed by the executing threads are complete. For example, when N is 0, the executing thread waits on all the prior bulk async-groups to complete. Operand N is an integer constant. 

By default, cp.async.bulk.wait_group instruction will cause the executing thread to wait until completion of all the bulk async operations in the specified bulk async-group. A bulk async operation includes the following: 

▶ Optionally, reading from the tensormap. 

▶ Reading from the source locations. 

▶ Writing to their respective destination locations. 

▶ Writes being made visible to the executing thread. 

The optional .read modifier indicates that the waiting has to be done until all the bulk async operations in the specified bulk async-group have completed: 

1. reading from the tensormap 

2. the reading from their source locations. 

# PTX ISA Notes

Introduced in PTX ISA version 8.0. 

# Target ISA Notes

Requires sm_90 or higher. 

# Examples

cp.async.bulk.wait_group.read 0; 

cp.async.bulk.wait_group 2; 

# 9.7.9.26 Data Movement and Conversion Instructions: tensormap.replace

# tensormap.replace

Modifies the field of a tensor-map object. 

# Syntax

```ruby
tensormap.replace_mode.field1{.ss}.b1024.type [addr], new_val;  
tensormap.replace_mode.field2{.ss}.b1024.type [addr], ord, new_val;  
tensormap.replace_mode.field3{.ss}.b1024.type [addr], new_val;  
模式 = { .tile }  
.field1 = { .global_address, .rank }  
.field2 = { .box_dim, .global_dim, .global_stride, .element_stride }  
.field3 = { .elemtype, .interleave.layout, .swizzle_mode, .swizzle_atomicity, .fill\_mode }  
.ss = { .global, .shared::cta }  
.type = { .b32, .b64 }
```

# Description

The tensormap.replace instruction replaces the field, specified by .field qualifier, of the tensormap object at the location specified by the address operand addr with a new value. The new value is specified by the argument new_val. 

Qualifier .mode specifies the mode of the tensor-map object located at the address operand addr. 

Instruction type .b1024 indicates the size of the tensor-map object, which is 1024 bits. 

Operand new_val has the type .type. When .field is specified as .global_address or . global_stride, .type must be .b64. Otherwise, .type must be .b32. 

The immediate integer operand ord specifies the ordinal of the field across the rank of the tensor which needs to be replaced in the tensor-map object. 

For field .rank, the operand new_val must be ones less than the desired tensor rank as this field uses zero-based numbering. 

When .field3 is specified, the operand new_val must be an immediate and the Table 33 shows the mapping of the operand new_val across various fields. 


Table 33: Tensormap new_val validity


<table><tr><td rowspan="2">new_val</td><td colspan="5">.field3</td></tr><tr><td>.elemtype</td><td>.inter-leave.layout</td><td>.swiz-zle_mode</td><td>.swiz-zle_atomicity</td><td>fill_mode</td></tr><tr><td>0</td><td>.u8</td><td>No interleave</td><td>No swizzling</td><td>16B</td><td>Zero fill</td></tr><tr><td>1</td><td>.u16</td><td>16B interleave</td><td>32B swiz-zling</td><td>32B</td><td>OOB-NaN fill</td></tr><tr><td>2</td><td>.u32</td><td>32B interleave</td><td>64B swiz-zling</td><td>32B + 8B flip</td><td>x</td></tr><tr><td>3</td><td>.s32</td><td>x</td><td>128B swiz-zling</td><td>64B</td><td>x</td></tr><tr><td>4</td><td>.u64</td><td>x</td><td>96B swiz-zling</td><td>x</td><td>x</td></tr><tr><td>5</td><td>.s64</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>6</td><td>.f16</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>7</td><td>.f32</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>8</td><td>.f32.ftz</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>9</td><td>.f64</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>10</td><td>.bf16</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>11</td><td>.tf32</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>12</td><td>.tf32.ftz</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>13</td><td>.b4x16</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>14</td><td>.b4x16_p64</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>15</td><td>.b6x16_p32 or .b6p2x16</td><td>x</td><td>x</td><td>x</td><td>x</td></tr></table>


Note: The values of .elemtype do not correspond to the values of the CUtensorMapDataType enum used in the driver API. 


If no state space is specified then Generic Addressing is used. If the address specified by addr does not fall within the address window of .global or .shared::cta state space then the behavior is undefined. 

tensormap.replace is treated as a weak memory operation, on the entire 1024-bit opaque tensormap object, in the Memory Consistency Model. 

# PTX ISA Notes

Introduced in PTX ISA version 8.3. 

Qualifier .swizzle_atomicity introduced in PTX ISA version 8.6. 

Qualifier .elemtype with values from 13 to 15, both inclusive, is supported in PTX ISA version 8.7 onwards. 

Qualifier .swizzle_mode with value 4 is supported from PTX ISA version 8.8 onwards. 

# Target ISA Notes

Supported on following architectures: 

▶ sm_90a 

▶ sm_100a 

▶ sm_101a (Renamed to sm_110a from PTX ISA version 9.0) 

▶ sm_120a 

▶ And is supported on following family-specific architectures from PTX ISA version 8.8: 

▶ sm_100f or higher in the same family 

▶ sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0) 

▶ sm_120f or higher in the same family 

▶ sm_110f or higher in the same family 

Qualifier .swizzle_atomicity is supported on following architectures: 

▶ sm_100a 

▶ sm_101a (Renamed to sm_110a from PTX ISA version 9.0) 

▶ sm_120a (refer to section for restrictions on sm_120a) 

▶ And is supported on following family-specific architectures from PTX ISA version 8.8: 

▶ sm_100f or higher in the same family 

▶ sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0) 

▶ sm_120f or higher in the same family 

▶ sm_110f or higher in the same family 

.field3 variant .elemtype corresponding to new_val values 13, 14 and 15 is supported on following architectures: 

▶ sm_100a 

▶ sm_101a (Renamed to sm_110a from PTX ISA version 9.0) 

▶ sm_120a (refer to section for restrictions on sm_120a) 

▶ And is supported on following family-specific architectures from PTX ISA version 8.8: 

▶ sm_100f or higher in the same family 

▶ sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0) 

▶ sm_120f or higher in the same family 

▶ sm_110f or higher in the same family 

.field3 variant .swizzle_mode corresponding to new_val value 4 is supported on following architectures: 

▶ sm_103a (refer to section for restrictions on sm_103a) 

# Examples

tensormap.replace.tile.global_address.shared::cta.b1024.b64 [sMem], new_val; 

# 9.7.10. Texture Instructions

This section describes PTX instructions for accessing textures and samplers. PTX supports the following operations on texture and sampler descriptors: 

▶ Static initialization of texture and sampler descriptors. 

▶ Module-scope and per-entry scope definitions of texture and sampler descriptors. 

▶ Ability to query fields within texture and sampler descriptors. 

# 9.7.10.1 Texturing Modes

For working with textures and samplers, PTX has two modes of operation. In the unified mode, texture and sampler information is accessed through a single .texref handle. In the independent mode, texture and sampler information each have their own handle, allowing them to be defined separately and combined at the site of usage in the program. 

The advantage of unified mode is that it allows 256 samplers per kernel (128 for architectures prior to sm_3x), with the restriction that they correspond 1-to-1 with the 256 possible textures per kernel (128 for architectures prior to sm_3x). The advantage of independent mode is that textures and samplers can be mixed and matched, but the number of samplers is greatly restricted to 32 per kernel (16 for architectures prior to sm_3x). 

Table 34 summarizes the number of textures, samplers and surfaces available in different texturing modes. 


Table 34: Texture, sampler and surface limits


<table><tr><td>Texturing mode</td><td>Resource</td><td>sm_1x, sm_2x</td><td>sm_3x+</td></tr><tr><td rowspan="3">Unified mode</td><td>Textures</td><td>128</td><td>256</td></tr><tr><td>Samplers</td><td>128</td><td>256</td></tr><tr><td>Surfaces</td><td>8</td><td>16</td></tr><tr><td rowspan="3">Independent mode</td><td>Textures</td><td>128</td><td>256</td></tr><tr><td>Samplers</td><td>16</td><td>32</td></tr><tr><td>Surfaces</td><td>8</td><td>16</td></tr></table>