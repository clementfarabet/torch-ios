//
//  ViewController.m
//  XOR_Example
//
//  Created by Kurt Jacobs on 2015/06/12.
//  Copyright (c) 2015 RandomDudes. All rights reserved.
//

#import "ViewController.h"

#define KBYTES_CLEAN_UP 10000 //10 Megabytes Max Storage Otherwise Force Cleanup (For This Example We Will Probably Never Reach It -- But Good Practice).
#define LUAT_STACK_INDEX_FLOAT_TENSORS 4 //Index of Garbage Collection Stack Value

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad
{
  [super viewDidLoad];
  
  UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(disableKeyboard)];
  [self.view addGestureRecognizer:tap];
  
  self.t = [Torch new];
  [self.t initialize];
  [self.t runMain:@"main" inFolder:@"xor_lua"];
}

- (void)disableKeyboard
{
  [self.valueTwoTextField resignFirstResponder];
  [self.valueOneTextfield resignFirstResponder];
}

- (IBAction)classifyAction:(id)sender
{
  if ([self isValidFloat:self.valueOneTextfield.text] && [self isValidFloat:self.valueTwoTextField.text])
  {
    float v1 = [self.valueOneTextfield.text floatValue];
    float v2 = [self.valueTwoTextField.text floatValue];
    [self perfClassificationOnValuesv1:v1 v2:v2];
  }
  else
  {
    self.answerLabel.text = @"Please Enter Valid Floats!!!";
  }
}

- (BOOL)isValidFloat:(NSString*)string
{
  NSScanner *scanner = [NSScanner scannerWithString:string];
  [scanner scanFloat:NULL];
  return [scanner isAtEnd];
}

- (void)perfClassificationOnValuesv1:(float)v1 v2:(float)v2
{
  XORClassifyObject *classificationObj = [XORClassifyObject new];
  classificationObj.x = v1;
  classificationObj.y = v2;
  float value = [self classifyExample:classificationObj inState:[self.t getLuaState]];
  self.answerLabel.text = [NSString stringWithFormat:@"Classification Value: %f",value];
}

- (CGFloat)classifyExample:(XORClassifyObject *)obj inState:(lua_State *)L
{
  NSInteger garbage_size_kbytes = lua_gc(L, LUA_GCCOUNT, LUAT_STACK_INDEX_FLOAT_TENSORS);

  if (garbage_size_kbytes >= KBYTES_CLEAN_UP)
  {
    NSLog(@"LUA -> Cleaning Up Garbage");
    lua_gc(L, LUA_GCCOLLECT, LUAT_STACK_INDEX_FLOAT_TENSORS);
  }

  THFloatStorage *classification_storage = THFloatStorage_newWithSize1(2);
  THFloatTensor *classification = THFloatTensor_newWithStorage1d(classification_storage, 1, 2, 1);
  THTensor_fastSet1d(classification, 0, obj.x);
  THTensor_fastSet1d(classification, 1, obj.y);
  lua_getglobal(L,"classifyExample");
  luaT_pushudata(L, classification, "torch.FloatTensor");
  
  //p_call -- args, results
  int res = lua_pcall(L, 1, 1, 0);
  if (res != 0)
  {
    NSLog(@"error running function `f': %s",lua_tostring(L, -1));
  }
  
  if (!lua_isnumber(L, -1))
  {
    NSLog(@"function `f' must return a number");
  }
  CGFloat returnValue = lua_tonumber(L, -1);
  lua_pop(L, 1);  /* pop returned value */
  return returnValue;
}

- (void)didReceiveMemoryWarning
{
  [super didReceiveMemoryWarning];
}

@end
