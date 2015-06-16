//
//  ViewController.m
//  XOR_Example
//
//  Created by Kurt Jacobs on 2015/06/12.
//  Copyright (c) 2015 RandomDudes. All rights reserved.
//

#import "ViewController.h"

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
  lua_getglobal(L,"classifyExample");
  THFloatTensor classify = *THFloatTensor_new();
  classify = *THFloatTensor_newWithStorage1d(classify.storage, 1, 2, 1);
  THFloatTensor_set1d(&classify, 0, obj.x);
  THFloatTensor_set1d(&classify, 1, obj.y);
  luaT_pushudata(L, &classify, "torch.FloatTensor");
  
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
